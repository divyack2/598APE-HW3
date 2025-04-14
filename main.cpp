#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include <sys/time.h>
#include <omp.h>

#include <immintrin.h>

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

struct PlanetArray{
   double *mass;
   double *x, *y;
   double *vx, *vy;
};

unsigned long long seed = 100;

unsigned long long randomU64() {
  seed ^= (seed << 21);
  seed ^= (seed >> 35);
  seed ^= (seed << 4);
  return seed;
}

double randomDouble()
{
   unsigned long long next = randomU64();
   next >>= 38;
   unsigned long long next2 = randomU64();
   next2 >>= 38;
   return ((next << 27) + next2) / (double)(1LL << 53);
}

int nplanets;
int timesteps;
double dt;
double G;

void next(const PlanetArray* planets, PlanetArray* nextplanets) {
   #pragma omp parallel for
   for (int i = 0; i < nplanets; i++) {
      double xi = planets->x[i];
      double yi = planets->y[i];
      double mi = planets->mass[i];
      double vxi = planets->vx[i];
      double vyi = planets->vy[i];

      // broadcasting into SIMD
      __m256d xi_vec = _mm256_set1_pd(xi);
      __m256d yi_vec = _mm256_set1_pd(yi);
      __m256d mi_vec = _mm256_set1_pd(mi);
      __m256d dt_vec = _mm256_set1_pd(dt);
      __m256d fx_vec = _mm256_setzero_pd();
      __m256d fy_vec = _mm256_setzero_pd();

      double fx = 0.0;
      double fy = 0.0;

      int j;
      for (j = 0; j <= nplanets - 4; j += 4) {
         // loading from memory
         __m256d xj = _mm256_loadu_pd(&planets->x[j]);
         __m256d yj = _mm256_loadu_pd(&planets->y[j]);
         __m256d mj = _mm256_loadu_pd(&planets->mass[j]);

         __m256d dx = _mm256_sub_pd(xj, xi_vec);
         __m256d dy = _mm256_sub_pd(yj, yi_vec);
         __m256d dx2 = _mm256_mul_pd(dx, dx);
         __m256d dy2 = _mm256_mul_pd(dy, dy);
         __m256d distSqr = _mm256_add_pd(_mm256_add_pd(dx2, dy2), _mm256_set1_pd(0.0001));
         __m256d dist = _mm256_sqrt_pd(distSqr);

         __m256d mprod = _mm256_mul_pd(mi_vec, mj);
         __m256d dist3 = _mm256_mul_pd(dist, _mm256_mul_pd(dist, dist));
         __m256d inv = _mm256_div_pd(mprod, dist3);
         __m256d val = _mm256_mul_pd(dt_vec, inv);

         // same as fx += dx * val
         fx_vec = _mm256_fmadd_pd(dx, val, fx_vec);
         fy_vec = _mm256_fmadd_pd(dy, val, fy_vec);
      }

      // get fx and fy from 4-element vectors
      double fx_arr[4], fy_arr[4];
      _mm256_storeu_pd(fx_arr, fx_vec);
      _mm256_storeu_pd(fy_arr, fy_vec);
      fx = fx_arr[0] + fx_arr[1] + fx_arr[2] + fx_arr[3];
      fy = fy_arr[0] + fy_arr[1] + fy_arr[2] + fy_arr[3];

      // handling tail case
      for (; j < nplanets; j++) {
         double dx = planets->x[j] - xi;
         double dy = planets->y[j] - yi;
         double distSqr = dx*dx + dy*dy + 0.0001;
         double dist = sqrt(distSqr);
         double val = dt * mi * planets->mass[j] / (dist * dist * dist);
         fx += dx * val;
         fy += dy * val;
      }

      double nvx = vxi + fx;
      double nvy = vyi + fy;
      nextplanets->vx[i] = nvx;
      nextplanets->vy[i] = nvy;
      nextplanets->x[i] = xi + dt * nvx;
      nextplanets->y[i] = yi + dt * nvy;
      nextplanets->mass[i] = mi;
   }
}


int main(int argc, const char** argv){
   if (argc < 2) {
      printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
      return 1;
   }
   nplanets = atoi(argv[1]);
   timesteps = atoi(argv[2]);
   dt = 0.001;
   G = 6.6743;

   // Set up SoA
   PlanetArray planets;
   planets.mass = (double*)malloc(sizeof(double) * nplanets);
   planets.x = (double*)malloc(sizeof(double) * nplanets);
   planets.y = (double*)malloc(sizeof(double) * nplanets);
   planets.vx = (double*)malloc(sizeof(double) * nplanets);
   planets.vy = (double*)malloc(sizeof(double) * nplanets);

   PlanetArray nextplanets;
   nextplanets.mass = (double*)malloc(sizeof(double) * nplanets);
   nextplanets.x = (double*)malloc(sizeof(double) * nplanets);
   nextplanets.y = (double*)malloc(sizeof(double) * nplanets);
   nextplanets.vx = (double*)malloc(sizeof(double) * nplanets);
   nextplanets.vy = (double*)malloc(sizeof(double) * nplanets);
   
   for (int i=0; i<nplanets; i++) {
      planets.mass[i] = randomDouble() * 10 + 0.2;
      planets.x[i] = ( randomDouble() - 0.5 ) * 100 * pow(1 + nplanets, 0.4);
      planets.y[i] = ( randomDouble() - 0.5 ) * 100 * pow(1 + nplanets, 0.4);
      planets.vx[i] = randomDouble() * 5 - 2.5;
      planets.vy[i] = randomDouble() * 5 - 2.5;
   }

   struct timeval start, end;
   gettimeofday(&start, NULL);
   for (int i=0; i<timesteps; i++) {
      // planets = next(planets);
      next(&planets, &nextplanets);
      PlanetArray temp = planets;
      planets = nextplanets;
      nextplanets = temp;
   }
   gettimeofday(&end, NULL);
   printf("Total time to run simulation %0.6f seconds, final location %f %f\n", tdiff(&start, &end), planets.x[nplanets-1], planets.y[nplanets-1]);

   // Free up memory
   free(planets.mass);
   free(planets.x);
   free(planets.y);
   free(planets.vx);
   free(planets.vy);

   free(nextplanets.mass);
   free(nextplanets.x);
   free(nextplanets.y);
   free(nextplanets.vx);
   free(nextplanets.vy);

   return 0;   
}