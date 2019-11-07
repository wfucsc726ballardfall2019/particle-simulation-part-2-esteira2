#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <iostream>
#include "common.h"
#include "omp.h"

using namespace std;

//
//  benchmarking program
//
int main( int argc, char **argv )
{   
    int navg,nabsavg=0,numthreads; 
    double dmin, absmin=1.0,davg,absavg=0.0;
	
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-p <int> to set number of threads\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    int p = read_int( argc, argv, "-p", 1 );
    omp_set_num_threads(p);

    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;      

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

    // Set the size of the bins
    // The grid size is sqrt(0.0005 * number of particles)
    // So make sure the bins only consider the cutoff radius where the particles actually react to each other
    double bin_length = getCutoff() * 2;
    int num_bins = ceil((sqrt(getDensity() * n)) / bin_length);

    // Array to keep track of which particles are in which bins
    vector<vector<particle_t> > bins(num_bins * num_bins);
    int offset_x;
    int offset_y;
    int which_bin;

    // Initialize the locks on each element in the bin
    /*omp_lock_t binlock[num_bins * num_bins];
    for (int i = 0; i < num_bins * num_bins; i++) {
        omp_init_lock(&(binlock[i]));
    }*/

    // We can just split up the particles among all the processors, and they can perform this computation independently
    // We want to make sure all threads can access the bins vector
    #pragma omp parallel for shared(bins)//, binlock) // This pragma divides the particles among the threads
    for (int i = 0; i < n; i++)
    {
        // Compute which bin a particle belongs to based on its location
        offset_x = floor(particles[i].x / bin_length);
        offset_y = floor(particles[i].y / bin_length);

        which_bin = num_bins * offset_y + offset_x;

        //cout << "here " << i << " which bin " << which_bin << endl;

        // Add the particle to the list of particles in that bin
        // We have to be careful here to avoid data races, so put a barrier here 
        //#pragma omp critical
        // Instead of putting a critical section (in which case the code will run serially), put a lock
        //omp_set_lock(&(binlock[which_bin]));
        //cout << "inside lock, particle " << i << ", which bin " << which_bin << endl;
        #pragma omp critical
        bins[which_bin].push_back(particles[i]);
        //omp_unset_lock(&(binlock[which_bin]));
    }

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );

    #pragma omp parallel private(dmin) shared(bins)//, binlock) // This pragma spawns the threads, so do all the parallel work in here
    {
    numthreads = p;

    // Make sure each thread has its own local copy of the following variables
    int thread_id = omp_get_thread_num();
    int offset_x;
    int offset_y;
    int which_bin;

    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        davg = 0.0;
	    dmin = 1.0;
        //
        //  compute all forces
        //

        // Consider each particle
        #pragma omp for reduction (+:navg) reduction(+:davg)
        for( int i = 0; i < n; i++ )
        {
            // Get the neighbors in a better way than a bunch of if-else statements

            particles[i].ax = particles[i].ay = 0;

            // Get the bin of the current particle
            offset_x = floor(particles[i].x / bin_length);
            offset_y = floor(particles[i].y / bin_length);

            //which_bin = num_bins * offset_y + offset_x;

            // Make sure the x position doesn't go beyond 0 to num_bins - 1
            for (int x = max(0, offset_x - 1); x <= min(offset_x + 1, num_bins - 1); x++) {
                // Make sure the y position doesn't go beyond 0 to num_bins - 1
                for (int y = max(0, offset_y - 1); y <= min(offset_y + 1, num_bins - 1); y++) {
                    // Now compute which bin we are currently considering for our force computation
                    which_bin = num_bins * y + x;
                    // Consider each particle in that bin 
                    for (int p = 0; p < bins[which_bin].size(); p++) {
                        // Compute the force between the current particle and the particles in this bin
                        apply_force(particles[i], bins[which_bin][p], &dmin, &davg, &navg);
                    }
                }
            }
        }
		
        //
        //  move particles
        //
        #pragma omp for
        for( int i = 0; i < n; i++ ) 
            move( particles[i] );
  
        if( find_option( argc, argv, "-no" ) == -1 ) 
        {
          //
          //  compute statistical data
          //
          #pragma omp master
          if (navg) { 
            absavg += davg/navg;
            nabsavg++;
          }

          #pragma omp critical
	      if (dmin < absmin) absmin = dmin; 
		
          //
          //  save if necessary
          //
          #pragma omp master
          if( fsave && (step%SAVEFREQ) == 0 )
              save( fsave, n, particles );
        }

        // The particles have moved, so update the particles in each bin
        // First, clear the current bin information
        #pragma omp for
        for (int i = 0; i < num_bins * num_bins; i++) 
            bins[i].resize(0);

        // Update
        #pragma omp for
        for (int i = 0; i < n; i++)
        {
            // Compute which bin a particle belongs to based on its location
            offset_x = floor(particles[i].x / bin_length);
            offset_y = floor(particles[i].y / bin_length);

            which_bin = num_bins * offset_y + offset_x;

            // Add the particle to the list of particles in that bin
            //#pragma omp critical
            // Instead of putting a critical section (in which case the code will run serially), put a lock
            //omp_set_lock(&(binlock[which_bin]));
            #pragma omp critical
            bins[which_bin].push_back(particles[i]);
            //omp_unset_lock(&(binlock[which_bin]));
        }
    }
    }
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

    // Destroy the locks
    /*for (int i = 0; i < num_bins * num_bins; i++) {
        omp_destroy_lock(&(binlock[i]));
    }*/

    if( find_option( argc, argv, "-no" ) == -1 ) 
    {
      if (nabsavg) absavg /= nabsavg;
    // 
    //  -The minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");
    
    //
    // Printing summary data
    //
    if( fsum)
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );

    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}

