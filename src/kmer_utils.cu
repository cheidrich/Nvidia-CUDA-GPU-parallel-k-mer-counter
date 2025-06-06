/* kmer_utils.cu
 *
 * Helper functions for k-mer key calculation and printing.
 */

#include "globals.cu"
#include <cmath>
#include <cstdio>
#include <cstdint>

/* DEVICE: compute numeric key from base sequence of length k */
__device__ uint64_t calcKmerKey(char seq[]) {
    uint64_t cnt=0, power=1; int tmp;
    for(size_t i=0;i<k;i++){
        char c=seq[i];
        switch(c){
            case 'A':case 'a':tmp=0;break;
            case 'C':case 'c':tmp=1;break;
            case 'G':case 'g':tmp=2;break;
            case 'T':case 't':tmp=3;break;
            default:return 0;
        }
        cnt+=tmp*power; power*=4;
    }
    return cnt;
}

/* HOST: recover k-mer string from numeric key */
void calcKmerFromKey(char* out, uint64_t key){
    const char b[]="ACGT";
    for(int i=0;i<khost;i++){
        out[i]=b[key%4];
        key/=4;
    }
    out[khost]='\0';
}

/* HOST: print first n k-mers whose count>=threshold */
void printKmerCounts(uint64_t keys[], uint64_t vals[],
                     size_t n, int threshold){
    char* buf=new char[khost+1];
    printf("Top %zu %d-mers >=%d:\n",n,khost,threshold);
    for(size_t i=0;i<n;i++){
        if((int)vals[i]>=threshold){
            calcKmerFromKey(buf,keys[i]);
            printf("%s : %llu\n",buf,(unsigned long long)vals[i]+1);
        }
    }
    delete[] buf;
}
