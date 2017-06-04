/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 50;
    weights.resize((unsigned long) num_particles);
    particles.resize((unsigned long) num_particles);

    random_device rd;
    default_random_engine gen(rd());
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {

        struct Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.f;

        particles[i] = p;
        weights[i] = (p.weight);
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	 for (int i = 0; i < num_particles; i++) {

        Particle *p = &particles[i];
        double x_new = p->x + velocity / yaw_rate * (sin(p->theta + yaw_rate * delta_t) - sin(p->theta));
        double y_new = p->y + velocity / yaw_rate * (cos(p->theta) - cos(p->theta + yaw_rate * delta_t));
        double theta_new = p->theta + yaw_rate * delta_t;

        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> dist_x(x_new, std_pos[0]);
        normal_distribution<double> dist_y(y_new, std_pos[1]);
        normal_distribution<double> dist_theta(theta_new, std_pos[2]);

        p->x = dist_x(gen);
        p->y = dist_y(gen);
        p->theta = dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
}
	LandmarkObs tranformObs(Particle particle, LandmarkObs observation) {
    LandmarkObs transformed_obs;

    transformed_obs.id = observation.id;
    transformed_obs.x = particle.x + (observation.x * cos(particle.theta)) - (observation.y * sin(particle.theta));
    transformed_obs.y = particle.y + (observation.x * sin(particle.theta)) + (observation.y * cos(particle.theta));

    return transformed_obs;

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
	double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double sum_weights = 0.;

    for (int p_i = 0; p_i < num_particles; p_i++) {
        Particle *p = &particles[p_i];
        double weight = 1.;

        for (int obs_i = 0; obs_i < observations.size(); obs_i++) {
            LandmarkObs obs = observations[obs_i];
            obs = tranformObs(*p, obs);

            Map::single_landmark_s best_lm;
            double shortest_dist = numeric_limits<double>::max();
            for (int lm_i = 0; lm_i < map_landmarks.landmark_list.size(); lm_i++) {
                Map::single_landmark_s lm = map_landmarks.landmark_list[lm_i];
                double cur_dist = dist(obs.x, obs.y, lm.x_f, lm.y_f);
                if (cur_dist < shortest_dist) {
                    shortest_dist = cur_dist;
                    best_lm = lm;
                }
            }

            double numerator = exp(-0.5 *
                                   (pow((obs.x - best_lm.x_f), 2) / pow(std_x, 2) +
                                    pow((obs.y - best_lm.y_f), 2) / pow(std_y, 2)));
            double denominator = 2 * M_PI * std_x * std_y;
            weight *= numerator / denominator;
        }

        sum_weights += weight;
        p->weight = weight;
    }

    for (int i = 0; i < num_particles; i++) {
        Particle *p = &particles[i];
        p->weight /= sum_weights;
        weights[i] = p->weight;
    }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	 random_device rd;
    default_random_engine gen(rd());

    discrete_distribution<> dist_particles(weights.begin(), weights.end());
    vector<Particle> resampled_particles((unsigned long) num_particles);

    for (int i = 0; i < num_particles; i++) {
        resampled_particles[i] = particles[dist_particles(gen)];
    }

    particles = resampled_particles;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
