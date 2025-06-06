/* kernels_fq.cu
 *
 * FASTQ kernels: countReadsFQ, findReadPositionsFQ, countKmersFQ
 */

#include "globals.cu"
#include "kmer_utils.cu"
#include <cooperative_groups.h>
#include "warpcore/include/warpcore.cuh"

using namespace cooperative_groups;

/* count valid '@' markers */
__global__ void countReadsFQ(char file[], uint64_t* size, int* out){
    uint64_t tid=threadIdx.x+blockIdx.x*blockDim.x,
             thr=blockDim.x*gridDim.x;
    uint64_t chunk=*size/thr, start=tid*chunk;
    if(tid<(*size%thr)){chunk++; start+=tid;}
    else start+=(*size%thr);

    int c=0;
    for(uint64_t i=0;i<chunk;i++){
        if(file[start+i]=='@'){
            if(start+i>0 && file[start+i-1]=='\n'){
                while(file[start+i]!='\n') i++;
                if(file[start+i+1] && file[start+i+1]!='@') c++;
            } else if(start+i==0) c++;
        }
    }
    atomicAdd(out,c);
}

/* record offsets of valid '@' */
__global__ void findReadPositionsFQ(char file[], uint64_t pos[], uint64_t* size){
    uint64_t tid=threadIdx.x+blockIdx.x*blockDim.x,
             thr=blockDim.x*gridDim.x;
    uint64_t chunk=*size/thr, start=tid*chunk;
    if(tid<(*size%thr)){chunk++; start+=tid;}
    else start+=(*size%thr);

    for(uint64_t i=0;i<chunk;i++){
        if(file[start+i]=='@'){
            if(start+i>0 && file[start+i-1]=='\n'){
                while(file[start+i]!='\n') i++;
                if(file[start+i+1] && file[start+i+1]!='@'){
                    int idx=atomicAdd(&arrayPosition,1);
                    atomicCAS(&pos[idx],pos[idx],start+i+1);
                }
            } else {
                int idx=atomicAdd(&arrayPosition,1);
                atomicCAS(&pos[idx],pos[idx],start+i);
            }
        }
    }
}

/* extract k-mers and insert into table+filter */
__global__ void countKmersFQ(char file[], uint64_t pos[], int* nreads,
    warpcore::CountingHashTable<uint64_t,uint64_t> cht,
    warpcore::BloomFilter<uint64_t,warpcore::hashers::MurmurHash<uint64_t>,uint64_t> bf)
{
    using namespace warpcore;
    uint64_t tid=threadIdx.x+blockIdx.x*blockDim.x,
             thr=blockDim.x*gridDim.x;
    int total=*nreads;
    uint64_t chunk=total/thr, start=tid*chunk;
    if(tid<(total%thr)){chunk++; start+=tid;}
    else start+=(total%thr);

    auto cg1=cg::tiled_partition<cht.cg_size()>(this_thread_block());
    auto cg2=cg::tiled_partition<bf.cg_size()>(this_thread_block());
    uint32_t m1=1,m2=1,vm=1; bool v=false;
    bool a1=true,a2=true;

    char buf[31];
    uint64_t key;
    int i=0;
    while(a1){
        char* it=&file[pos[start+i]];
        while(*it!='\n') it++;
        it++;
        for(int j=1;j<k;j++){
            if(*it!='\n' && *it!=' ' && *it!='\r') buf[j]=*it;
            else j--;
            it++;
        }
        while(a2){
            v=false;
            if((*it=='A'||*it=='C'||*it=='G'||*it=='T')&&a1&&a2){
                for(int j=0;j<k;j++){
                    buf[j]=(j==k-1?*it:buf[j+1]);
                }
                key=calcKmerKey(buf);
                v=bf.insert_and_query(key,cg2);
            }
            vm=cg1.ballot(v);
            while(vm){
                int l=__ffs(vm)-1;
                uint64_t fk=cg1.shfl(key,l);
                cht.insert(fk,cg1);
                vm^=1u<<l;
            }
            if(*it=='+'||!*it) a2=false;
            else it++;
            m2=cg1.ballot(a2);
        }
        a2=true; i++;
        if(i>=chunk) a1=false;
        m1=cg1.ballot(a1);
    }
}
