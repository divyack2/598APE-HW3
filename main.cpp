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

// Planet* next(Planet* planets) {
//    Planet* nextplanets = (Planet*)malloc(sizeof(Planet) * nplanets);
//    #pragma omp parallel for simd
//    for (int i=0; i<nplanets; i++) {
//       nextplanets[i].vx = planets[i].vx;
//       nextplanets[i].vy = planets[i].vy;
//       nextplanets[i].mass = planets[i].mass;
//       nextplanets[i].x = planets[i].x;
//       nextplanets[i].y = planets[i].y;
//    }

//    #pragma omp parallel for simd
//    for (int i=0; i<nplanets; i++) {
//       for (int j=0; j<nplanets; j++) {
//          double dx = planets[j].x - planets[i].x;
//          double dy = planets[j].y - planets[i].y;
//          double distSqr = dx*dx + dy*dy + 0.0001;
//          double invDist = planets[i].mass * planets[j].mass / sqrt(distSqr);
//          double invDist3 = invDist * invDist * invDist;
//          double val = dt * invDist3;
//          nextplanets[i].vx += dx * val;
//          nextplanets[i].vy += dy * val;
//       }
//       nextplanets[i].x += dt * nextplanets[i].vx;
//       nextplanets[i].y += dt * nextplanets[i].vy;
//    }
//    free(planets);
//    return nextplanets;
// }

// PlanetArray next(PlanetArray planets) {
//    // Set up nextplanets SoA
//    PlanetArray nextplanets;
//    nextplanets.mass = (double*)malloc(sizeof(double) * nplanets);
//    nextplanets.x = (double*)malloc(sizeof(double) * nplanets);
//    nextplanets.y = (double*)malloc(sizeof(double) * nplanets);
//    nextplanets.vx = (double*)malloc(sizeof(double) * nplanets);
//    nextplanets.vy = (double*)malloc(sizeof(double) * nplanets);

//    #pragma omp parallel for
//    for (int i=0; i<nplanets; i++) {
//       nextplanets.vx[i] = planets.vx[i];
//       nextplanets.vy[i] = planets.vy[i];
//       nextplanets.mass[i] = planets.mass[i];
//       nextplanets.x[i] = planets.x[i];
//       nextplanets.y[i] = planets.y[i];

//       for (int j=0; j<nplanets; j++) {
//          double dx = planets.x[j] - planets.x[i];
//          double dy = planets.y[j] - planets.y[i];
//          double distSqr = dx*dx + dy*dy + 0.0001;
//          double invDist = planets.mass[i] * planets.mass[j] / sqrt(distSqr);
//          double invDist3 = invDist * invDist * invDist;
//          double val = dt * invDist3;
//          nextplanets.vx[i] += dx * val;
//          nextplanets.vy[i] += dy * val;
//       }
//       nextplanets.x[i] += dt * nextplanets.vx[i];
//       nextplanets.y[i] += dt * nextplanets.vy[i];
//    }
      
//    free(planets.mass);
//    free(planets.x);
//    free(planets.y);
//    free(planets.vx);
//    free(planets.vy);
//    return nextplanets;
// }

void next(const PlanetArray* planets, PlanetArray* nextplanets) {
   #pragma omp parallel for simd
   for (int i=0; i<nplanets; i++) {
      nextplanets->vx[i] = planets->vx[i];
      nextplanets->vy[i] = planets->vy[i];
      nextplanets->mass[i] = planets->mass[i];
      nextplanets->x[i] = planets->x[i];
      nextplanets->y[i] = planets->y[i];

      for (int j=0; j<nplanets; j++) {
         double dx = planets->x[j] - planets->x[i];
         double dy = planets->y[j] - planets->y[i];
         double distSqr = dx*dx + dy*dy + 0.0001;
         double invDist = planets->mass[i] * planets->mass[j] / sqrt(distSqr);
         double invDist3 = invDist * invDist * invDist;
         double val = dt * invDist3;
         nextplanets->vx[i] += dx * val;
         nextplanets->vy[i] += dy * val;
      }
      nextplanets->x[i] += dt * nextplanets->vx[i];
      nextplanets->y[i] += dt * nextplanets->vy[i];
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