#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include <sys/time.h>
#include <omp.h>

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

struct Planet {
   double mass;
   double x;
   double y;
   double vx;
   double vy;
};

unsigned long long seed = 100;

unsigned long long randomU64() {
  seed ^= (seed << 21);
  seed ^= (seed >> 35);
  seed ^= (seed << 4);
  return seed;
}

// TODO: possibly find a faster way of computing a random number
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

Planet* next(Planet* planets) {
   Planet* nextplanets = (Planet*)malloc(sizeof(Planet) * nplanets);
   #pragma omp parallel for simd
   for (int i=0; i<nplanets; i++) {
      nextplanets[i].vx = planets[i].vx;
      nextplanets[i].vy = planets[i].vy;
      nextplanets[i].mass = planets[i].mass;
      nextplanets[i].x = planets[i].x;
      nextplanets[i].y = planets[i].y;
   }

   #pragma omp parallel for simd
   for (int i=0; i<nplanets; i++) {
      for (int j=0; j<nplanets; j++) {
         double dx = planets[j].x - planets[i].x;
         double dy = planets[j].y - planets[i].y;
         double distSqr = dx*dx + dy*dy + 0.0001;
         double invDist = planets[i].mass * planets[j].mass / sqrt(distSqr);
         double invDist3 = invDist * invDist * invDist;
         double val = dt * invDist3;
         nextplanets[i].vx += dx * val;
         nextplanets[i].vy += dy * val;
      }
      nextplanets[i].x += dt * nextplanets[i].vx;
      nextplanets[i].y += dt * nextplanets[i].vy;
   }
   free(planets);
   return nextplanets;
}

Planet* serial_next(Planet* planets) {
   Planet* nextplanets = (Planet*)malloc(sizeof(Planet) * nplanets);
   for (int i=0; i<nplanets; i++) {
      nextplanets[i].vx = planets[i].vx;
      nextplanets[i].vy = planets[i].vy;
      nextplanets[i].mass = planets[i].mass;
      nextplanets[i].x = planets[i].x;
      nextplanets[i].y = planets[i].y;
   }

   for (int i=0; i<nplanets; i++) {
      for (int j=0; j<nplanets; j++) {
         double dx = planets[j].x - planets[i].x;
         double dy = planets[j].y - planets[i].y;
         double distSqr = dx*dx + dy*dy + 0.0001;
         double invDist = planets[i].mass * planets[j].mass / sqrt(distSqr);
         double invDist3 = invDist * invDist * invDist;
         double val = dt * invDist3;
         nextplanets[i].vx += dx * val;
         nextplanets[i].vy += dy * val;
      }
      nextplanets[i].x += dt * nextplanets[i].vx;
      nextplanets[i].y += dt * nextplanets[i].vy;
   }
   free(planets);
   return nextplanets;
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

   Planet* planets = (Planet*)malloc(sizeof(Planet) * nplanets);
   double random = randomDouble();
   if (nplanets > 50) {
      #pragma omp parallel for
      for (int i=0; i<nplanets; i++) {
         planets[i].mass = random * 10 + 0.2;
         double val1 = ( random - 0.5 ) * 100 * pow(1 + nplanets, 0.4);
         planets[i].x = val1;
         planets[i].y = val1;
         double val2 = random * 5 - 2.5;
         planets[i].vx = val2;
         planets[i].vy = val2;
      }
   } else {
      for (int i=0; i<nplanets; i++) {
         planets[i].mass = random * 10 + 0.2;
         double val1 = ( random - 0.5 ) * 100 * pow(1 + nplanets, 0.4);
         planets[i].x = val1;
         planets[i].y = val1;
         double val2 = random * 5 - 2.5;
         planets[i].vx = val2;
         planets[i].vy = val2;
      }
   }

   struct timeval start, end;
   gettimeofday(&start, NULL);
   for (int i=0; i<timesteps; i++) {
      // if (i % 10000000 == 0) {
      //    printf("At iteration %d \n", i);
      // }
      if (nplanets > 50) {
         planets = next(planets);
      } else {
         planets = serial_next(planets);
      }
      
   }
   gettimeofday(&end, NULL);
   printf("Total time to run simulation %0.6f seconds, final location %f %f\n", tdiff(&start, &end), planets[nplanets-1].x, planets[nplanets-1].y);

   return 0;   
}