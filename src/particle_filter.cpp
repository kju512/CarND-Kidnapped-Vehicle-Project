/*
 * particle_filter.cpp
 *
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
        std::default_random_engine gen;
        double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
        std_x=std[0];
	std_y=std[1];
	std_theta=std[2];	
	
	//creates a Gaussian distribution
	std::normal_distribution<double> dist_x(x, std_x);
	std::normal_distribution<double> dist_y(y, std_y);
	std::normal_distribution<double> dist_theta(theta, std_theta);
        //creates all particles
        for(int i=0;i<num_particles;i++){
           Particle sample_particle;
           sample_particle.id=i;
           sample_particle.x=dist_x(gen);
           sample_particle.y=dist_y(gen);
	   sample_particle.theta=dist_theta(gen);
           sample_particle.weight=1.0;
           particles.push_back(sample_particle);
	   weights.push_back(1.0);
        }
        is_initialized =true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
        std::default_random_engine gen;
        double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
        std_x=std_pos[0];
	std_y=std_pos[1];
	std_theta=std_pos[2];	
	
	//creates a Gaussian distribution
	std::normal_distribution<double> dist_x(0, std_x);
	std::normal_distribution<double> dist_y(0, std_y);
	std::normal_distribution<double> dist_theta(0, std_theta);
        for(int i=num_particles-1;i>=0;i--){
            Particle sample_particle=particles.at(i);
            double x,y,theta;
            //calculate the predicted position
            if(yaw_rate<0.001){
                x=sample_particle.x+velocity*cos(sample_particle.theta)*delta_t;
                y=sample_particle.y+velocity*sin(sample_particle.theta)*delta_t;
                theta=sample_particle.theta+yaw_rate*delta_t;
            }
            else{
                x=sample_particle.x+(velocity/yaw_rate)*(sin(sample_particle.theta+yaw_rate*delta_t)-sin(sample_particle.theta));
                y=sample_particle.y+(velocity/yaw_rate)*(cos(sample_particle.theta)-cos(sample_particle.theta+yaw_rate*delta_t));
                theta=sample_particle.theta+yaw_rate*delta_t;
            }
            //add random Gaussian noise
            x+=dist_x(gen);
            y+=dist_y(gen);
            theta+=dist_theta(gen);
            //write new position to current particle
            sample_particle.x=x;
            sample_particle.y=y;
	    sample_particle.theta=theta;
            particles.at(i)=sample_particle;
        }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
        std::vector<LandmarkObs> newobs;
        int flag;
        double min,dis;
        for(int i=0;i<predicted.size();i++){
            flag=0;
            min=50;
            for(int j=0;j<observations.size();j++){
                dis=dist(predicted.at(i).x,predicted.at(i).y,observations.at(j).x,observations.at(j).y);
                if(dis<min){
                    min=dis;
                    flag=j;
                }
            }
            newobs.push_back(observations.at(flag));
        }
        observations=newobs;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
        
        for(int k=0;k<num_particles;k++){
             std::vector<LandmarkObs> predicted;
             std::vector<LandmarkObs> newobs;
             for(int j=0;j<observations.size();j++){
                 double xt,yt,theta,x,y;
                 //get particle position
                 xt=particles.at(k).x;
                 yt=particles.at(k).y;
	         theta=particles.at(k).theta;
                 //get observation position coordinates
                 x=observations.at(j).x;
                 y=observations.at(j).y;
                 //transform observation position coordinates to map cordinates
                 transformObsToMap(&x,&y,xt,yt,theta);
                 LandmarkObs predlandmark;
                 predlandmark.id=0;
                 predlandmark.x=x;
                 predlandmark.y=y;
                 predicted.push_back(predlandmark);
             }
             for(int j=0;j<map_landmarks.landmark_list.size();j++){
                 LandmarkObs predlandmark;
                 double dis;
                 predlandmark.id=map_landmarks.landmark_list.at(j).id_i;
                 predlandmark.x=map_landmarks.landmark_list.at(j).x_f;
                 predlandmark.y=map_landmarks.landmark_list.at(j).y_f;
                 dis=dist(predlandmark.x,predlandmark.y,particles.at(k).x,particles.at(k).y);
                 if(dis<=sensor_range)
                     newobs.push_back(predlandmark);
             }             
             dataAssociation(predicted,newobs);
             double prob=1.0;
             for(int j=0;j<newobs.size();j++){
                 double x1,y1,x2,y2;
                 x1=predicted.at(j).x;
                 y1=predicted.at(j).y;
                 x2=newobs.at(j).x;
                 y2=newobs.at(j).y;
                 //calculate the particle's multi-variate Gaussian probility               
                 prob*=calculateGaussianDisValue(x1,y1,x2,y2,std_landmark[0],std_landmark[1]);

             }
             particles.at(k).weight=prob;
             weights.at(k)=prob;
        }    
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<int> p(weights.begin(), weights.end());
        std::vector<Particle> newparticles;
        for(int n=0; n<num_particles; ++n) {
           newparticles.push_back(particles.at(p(gen)));
        }
        particles=newparticles;
}

/**
* calculate the Gaussian distribution probobility value
*/
double ParticleFilter::calculateGaussianDisValue(double x,double y,double x_mean,double y_mean,double std_x,double std_y){
        
        return (1.0/2.0*M_PI/std_x/std_y)*exp(-(x-x_mean)*(x-x_mean)/2.0/std_x/std_x-(y-y_mean)*(y-y_mean)/2.0/std_y/std_y);
}
/**
* transform observation coordinate to the map coordinate
*/
void ParticleFilter::transformObsToMap(double* x,double* y,double x0,double y0,double theta){
       double tempx,tempy;
       tempx=*x*cos(theta)-*y*sin(theta)+x0;
       tempy=*x*sin(theta)+*y*cos(theta)+y0;
       *x=tempx;
       *y=tempy;
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

std::string ParticleFilter::getAssociations(Particle best)
{
	std::vector<int> v = best.associations;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseX(Particle best)
{
	std::vector<double> v = best.sense_x;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseY(Particle best)
{
	std::vector<double> v = best.sense_y;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
