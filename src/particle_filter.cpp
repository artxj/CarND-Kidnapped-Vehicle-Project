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
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	if (is_initialized) return;

	num_particles = 100;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (unsigned int i = 0; i < num_particles; ++i) {
		Particle particle;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
		weights.push_back(1.0);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	double coeff;
	if (fabs(yaw_rate) > 1e-2) {
		coeff = velocity / yaw_rate;
	}

	// noise gen
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (unsigned int i = 0; i < num_particles; ++i) {
		Particle particle = particles[i];

		// calculating new state
		if (fabs(yaw_rate) > 1e-2) {
			const double yaw_t = yaw_rate * delta_t;
			particle.x += coeff * (sin(particle.theta + yaw_t) - sin(particle.theta));
			particle.y += coeff * (cos(particle.theta) - cos(particle.theta + yaw_t));
			particle.theta += yaw_t;
		} else {
			const double velocity_t = velocity * delta_t;
			particle.x += velocity_t * cos(particle.theta);
			particle.y += velocity_t * sin(particle.theta);
		}

		// adding noise
		particle.x += dist_x(gen);
		particle.y += dist_y(gen);
		particle.theta += dist_theta(gen);

		// updating particle in vector
		particles[i] = particle;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); ++i) {
		double min_index = 0;
		double min_value = numeric_limits<double>::max();
		for (unsigned j = 0; j < predicted.size(); ++j) {
			const double distance = calculate_distance(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (distance < min_value) {
				min_value = distance;
				min_index = j;
			}
		}
		// NOTE: We set index of predicted observation as ID, not actual ID
		// it's done to maintain fast calculations later
		// if actual ID is needed some time later, both this code
		// and the code within updateWeights should be updated
		observations[i].id = min_index; // predicted[min_index].id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// mult-variate Gaussian distribution coefficients for bivariate case
	const double std_x = std_landmark[0];
	const double std_y = std_landmark[1];
	const double cov_coeff = 1.0 / (2 * M_PI * std_x * std_y);
	const double std_x_sq = 1.0 / (std_x * std_x);
	const double std_y_sq = 1.0 / (std_y * std_y);

	for (unsigned int i = 0; i < num_particles; ++i) {
		Particle particle = particles[i];

		// determing the list of possible landmarks based on given sensor range
		vector<LandmarkObs> possible_landmarks;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			const Map::single_landmark_s map_landmark = map_landmarks.landmark_list[j];
			if (calculate_distance(particle.x, particle.y, map_landmark.x_f, map_landmark.y_f) < 1.2 * sensor_range) {
				LandmarkObs landmark;
				landmark.id = map_landmark.id_i;
				landmark.x = map_landmark.x_f;
				landmark.y = map_landmark.y_f;
				possible_landmarks.push_back(landmark);
			}
		}

		// if no landmarks at all - skip this particle
		if (possible_landmarks.size() == 0) continue;

		// transforming the coordinates from vehicle to map system
		vector<LandmarkObs> transformed_observations;
		for (unsigned int j = 0; j < observations.size(); ++j) {
			LandmarkObs landmark = observations[j];

			const double sin_theta = sin(particle.theta);
			const double cos_theta = cos(particle.theta);
			LandmarkObs transformed;
			transformed.x = particle.x + cos_theta * landmark.x - sin_theta * landmark.y;
			transformed.y = particle.y + sin_theta * landmark.x + cos_theta * landmark.y;
			transformed_observations.push_back(transformed);
		}

		// associate the observations with landmarks
		dataAssociation(possible_landmarks, transformed_observations);

		// calculating weights
		double new_weight = 1;
		for (unsigned int i = 0; i < transformed_observations.size(); ++i) {
			const LandmarkObs observation = transformed_observations[i];
			const LandmarkObs predicted = possible_landmarks[observation.id];
			const double x_diff = observation.x - predicted.x;
			const double y_diff = observation.y - predicted.y;
			new_weight *= cov_coeff * exp( -0.5 * (std_x_sq * x_diff * x_diff + std_y_sq * y_diff * y_diff) );
		}
		particle.weight = new_weight;
		particles[i] = particle;
		weights[i] = new_weight;
	}

}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> sampled_particles;
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	for (unsigned int i = 0; i < num_particles; ++i) {
		sampled_particles.push_back(particles[distribution(gen)]);
	}
	particles = sampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " ") );
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " ") );
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " ") );
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
