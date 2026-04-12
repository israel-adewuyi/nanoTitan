#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

void init_comms(ncclComm_t* comms, int ndev, const int* devlist) {
    ncclUniqueId id;
    ncclGetUniqueId(&id);

    ncclGroupStart();
    for (int i = 0; i < ndev; i++) {
        cudaSetDevice(devlist[i]);
        ncclCommInitRank(&comms[i], ndev, id, i);
    }
    ncclGroupEnd();
}


int main(){
    int iter_warmup = 10;
    int iter_execute = 10000;
    int devices[2] = {0, 6};
    ncclComm_t comms[2];
    cudaStream_t streams[2];
    float *bufs[2];

    for(size_t bytes = 8; bytes <= (1ull<<20); bytes*=2){
        size_t count = bytes / sizeof(float);

        std::vector<float>ones_arr(count), twos_arr(count), out_arr(count);

        for(int i = 0; i < count; i++){
            ones_arr[i] = 1.0;
            twos_arr[i] = 2.0;
        }
        
        for(int i = 0; i < 2; i++){
            cudaSetDevice(devices[i]);
            cudaStreamCreate(&streams[i]);
            cudaMalloc(&bufs[i], count * sizeof(float));
            
            if(i == 0){
                cudaMemcpy(bufs[i], ones_arr.data(), count * sizeof(float), cudaMemcpyHostToDevice);
            }
            else{
                cudaMemcpy(bufs[i], twos_arr.data(), count * sizeof(float), cudaMemcpyHostToDevice);
            }
            
        }
    
        init_comms(comms, 2, devices);

        for(int iter = 0; iter < iter_warmup; iter++){
            ncclGroupStart();
            for(int i = 0; i < 2; i++){
                cudaSetDevice(devices[i]);
                ncclAllReduce(bufs[i], bufs[i], count, ncclFloat, ncclSum, comms[i], streams[i]);
            }

            ncclGroupEnd();

            for (int i = 0; i < 2; i++) {
                cudaSetDevice(devices[i]);
                cudaStreamSynchronize(streams[i]);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        for(int iter = 0; iter < iter_execute; iter++){
            ncclGroupStart();
            for(int i = 0; i < 2; i++){
                cudaSetDevice(devices[i]);
                ncclAllReduce(bufs[i], bufs[i], count, ncclFloat, ncclSum, comms[i], streams[i]);
            }

            ncclGroupEnd();

            for (int i = 0; i < 2; i++) {
                cudaSetDevice(devices[i]);
                cudaStreamSynchronize(streams[i]);
            }
        }

        for (int i = 0; i < 2; i++) {
            cudaSetDevice(devices[i]);
            cudaFree(bufs[i]);
            cudaStreamDestroy(streams[i]);
            ncclCommDestroy(comms[i]);
        }

        auto stop = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double avg_duration = (double)duration.count() / iter_execute;
        double avg_duration_s = avg_duration * 1e-6;
        double alg_bandw = (double)bytes / avg_duration_s / 1e9;

        // std::cout << "Latency(us): " << avg_duration << "\n";
        // std::cout << "AlgBW(GB/s): " << alg_bandw << "\n";
        std::cout << "Bytes = " << bytes << " , Latency: " << avg_duration << " ,Bandwidth: " << alg_bandw << "\n";
    }

    printf("AllReduce finished\n");
    return 0;
}
