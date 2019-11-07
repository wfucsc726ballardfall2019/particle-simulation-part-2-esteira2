#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <iostream>
#include "common.h"

using namespace std;

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank ); // Processor ID numbers
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE ); // Stores info related to a particle contiguously
    MPI_Type_commit( &PARTICLE );
    
    //
    //  set up the data partitioning across processors
    //
    int particle_per_proc = (n + n_proc - 1) / n_proc;
    int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) ); // Describes index into particle array for knowing which particles belong to which process
    for( int i = 0; i < n_proc+1; i++ )
        partition_offsets[i] = min( i * particle_per_proc, n );
    
    int *partition_sizes = (int*) malloc( n_proc * sizeof(int) ); // Says how many particles a process is responsible for
    for( int i = 0; i < n_proc; i++ )
        partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];

    //
    //  allocate storage for local partition
    //
    int nlocal = partition_sizes[rank];
    particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) ); // Create array to store the particles local to a process

    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );
    if( rank == 0 ) {
        init_particles( n, particles );
    }
    MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD ); // Now each processor has a chunk of the particles
    
    // Set the size of the bins
    // The grid size is sqrt(0.0005 * number of particles)
    // So make sure the bins only consider the cutoff radius where the particles actually react to each other
    double bin_length = getCutoff() * 2;
    int num_bins = ceil((sqrt(getDensity() * n)) / bin_length);

    // Array to keep track of which particles are in which bins
    // This will be a local copy for each processor
    vector<vector<particle_t> > bins(num_bins * num_bins);
    int offset_x;
    int offset_y;
    int which_bin;

    // Compute which particles belong in which bin in parallel
    //cout << "**** PROCESS " << rank << " ABOUT TO COMPUTE LOCAL BINS ****\n";
    for (int i = 0; i < nlocal; i++)
    {
        // Compute which bin a particle belongs to based on its location
        offset_x = floor(local[i].x / bin_length);
        offset_y = floor(local[i].y / bin_length);

        which_bin = num_bins * offset_y + offset_x;

        // Add the particle to the list of particles in that bin
        // This is the processors local version of the bins, so we don't have to worry about race conditions
        bins[which_bin].push_back(local[i]);
    }

    // Send all the bins info to processor 0, who will then consolidate that into one list
    for (int i = 0; i < bins.size(); i++) {
        if (rank != 0) {
            int data_amt = bins[i].size();
            MPI_Send(&data_amt, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            MPI_Send(&bins[i][0], data_amt, PARTICLE, 0, 0, MPI_COMM_WORLD);
        }
        else {
            // Processor 0 collects all the info
            for (int r = 1; r < n_proc; r++) {
                int data_amt;
                MPI_Recv(&data_amt, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //cout << "~~~ PROCESS " << rank << " is receiving data ~~~\n";
                vector<particle_t> temp_bini(data_amt);
                MPI_Recv(&temp_bini[0], data_amt, PARTICLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int j = 0; j < temp_bini.size(); j++) {
                    bins[i].push_back(temp_bini[j]);
                }
            }
        }
    }

    // Send from processor 0 the complete list of bins to all the processors
    // First, figure out how much data needs to be sent per bin
    int * bins_i_lengths = new (nothrow) int[num_bins * num_bins];
    if (rank == 0) {
        for (int i = 0; i < num_bins * num_bins; i++) {
            bins_i_lengths[i] = bins[i].size();
        }
    }

    // Next make sure each processor knows how much data to receive
    MPI_Bcast(bins_i_lengths, num_bins * num_bins, MPI_INT, 0, MPI_COMM_WORLD);

    // Make sure each processor has the correct amount of memory in place to receive data
    if (rank != 0){
        for (int i = 0; i < num_bins * num_bins; i++) {
            bins[i].resize(bins_i_lengths[i]);
        }
    }

    // Broadcast the bins
    for (int i = 0; i < num_bins * num_bins; i++) {
        MPI_Bcast(&bins[i][0], bins_i_lengths[i], PARTICLE, 0, MPI_COMM_WORLD);
    }

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;
        // 
        //  collect all global data locally (not good idea to do)
        //
        // Now all the processors have access to the particles array
        MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );
        
        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
          if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
        
        //
        //  compute all forces
        //

        // Loop through this processor's chunk of the particles
        for( int i = 0; i < nlocal; i++ )
        {
            local[i].ax = local[i].ay = 0;

            // Get the bin of the current particle
            offset_x = floor(local[i].x / bin_length);
            offset_y = floor(local[i].y / bin_length);

            // Make sure the x position doesn't go beyond 0 to num_bins - 1
            for (int x = max(0, offset_x - 1); x <= min(offset_x + 1, num_bins - 1); x++) {
                // Make sure the y position doesn't go beyond 0 to num_bins - 1
                for (int y = max(0, offset_y - 1); y <= min(offset_y + 1, num_bins - 1); y++) {
                    // Now compute which bin we are currently considering for our force computation
                    which_bin = num_bins * y + x;
                    // Consider each particle in that bin 
                    for (int p = 0; p < bins[which_bin].size(); p++) {
                        // Compute the force between the current particle and the particles in this bin
                        apply_force(local[i], bins[which_bin][p], &dmin, &davg, &navg);
                    }
                }
            }
        }
     
        if( find_option( argc, argv, "-no" ) == -1 )
        {
          
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

 
          if (rank == 0){
            //
            // Computing statistical data
            //
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }

        //
        //  move particles
        //
        for( int i = 0; i < nlocal; i++ )
            move( local[i] );


        // Update bins and re-broadcast to everyone
        // Clear current bin information
        for (int i = 0; i < num_bins * num_bins; i++) 
            bins[i].resize(0);
        
        // Update
        for (int i = 0; i < nlocal; i++)
        {
            // Compute which bin a particle belongs to based on its location
            offset_x = floor(local[i].x / bin_length);
            offset_y = floor(local[i].y / bin_length);

            which_bin = num_bins * offset_y + offset_x;

            // Add the particle to the list of particles in that bin
            // This is the processors local version of the bins, so we don't have to worry about race conditions
            bins[which_bin].push_back(local[i]);
        }

        for (int i = 0; i < bins.size(); i++) {
            if (rank != 0) {
                int data_amt = bins[i].size();
                MPI_Send(&data_amt, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

                MPI_Send(&bins[i][0], data_amt, PARTICLE, 0, 0, MPI_COMM_WORLD);
            }
            else {
                // Processor 0 collects all the info
                for (int r = 1; r < n_proc; r++) {
                    int data_amt;
                    MPI_Recv(&data_amt, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    vector<particle_t> temp_bini(data_amt);
                    MPI_Recv(&temp_bini[0], data_amt, PARTICLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    for (int j = 0; j < temp_bini.size(); j++) {
                        bins[i].push_back(temp_bini[j]);
                    }
                }
            }
        }

        // Send from processor 0 the complete list of bins to all the processors
        // First, figure out how much data needs to be sent per bin
        int * bins_i_lengths = new (nothrow) int[num_bins * num_bins];
        if (rank == 0) {
            for (int i = 0; i < num_bins * num_bins; i++) {
                bins_i_lengths[i] = bins[i].size();
            }
        }

        // Next make sure each processor knows how much data to receive
        MPI_Bcast(bins_i_lengths, num_bins * num_bins, MPI_INT, 0, MPI_COMM_WORLD);

        // Make sure each processor has the correct amount of memory in place to receive data
        if (rank != 0){
            for (int i = 0; i < num_bins * num_bins; i++) {
                bins[i].resize(bins_i_lengths[i]);
            }
        }

        // Broadcast the bins
        for (int i = 0; i < num_bins * num_bins; i++) {
            MPI_Bcast(&bins[i][0], bins_i_lengths[i], PARTICLE, 0, MPI_COMM_WORLD);
        }

    }
    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

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
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    free( partition_offsets );
    free( partition_sizes );
    free( local );
    free( particles );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
