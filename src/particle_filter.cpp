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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);


  for (int i=0; i<num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0f;
    particles.push_back(p);
  }
  weights.resize(particles.size(), 1.0f);
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);

  for (auto& particle : particles) {
    if (fabs(yaw_rate) < 0.001) {
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    } else {
      particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
      particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
      particle.theta += yaw_rate * delta_t;
    }
    particle.x += noise_x(gen);
    particle.y += noise_y(gen);
    particle.theta += noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
  if (predicted.empty()) return;
  for (auto& observation : observations) {
    auto it = predicted.begin();
    auto end = predicted.end();
    auto min = dist(it->x, it->y, observation.x, observation.y);
    observation.id = it->id;
    while (++it != end) {
      auto d = dist(it->x, it->y, observation.x, observation.y);
      if (d < min) {
        min = d;
        observation.id = it->id;
      }
    }
  }
}

LandmarkObs transformsCoordinates(Particle particle, LandmarkObs observation) {
  LandmarkObs ob;
  ob.x = particle.x + cos(particle.theta) * observation.x - sin(particle.theta) * observation.y;
  ob.y = particle.y + sin(particle.theta) * observation.x + cos(particle.theta) * observation.y;
  ob.id = observation.id;
  return ob;
}

double multivariateGaussianProb(LandmarkObs ob, LandmarkObs landmark, double std[]) {
  return exp(-(pow(ob.x-landmark.x,2)/pow(std[0],2)+pow(ob.y-landmark.y,2)/pow(std[1],2))/2)/(2.0f*M_PI*std[0]*std[1]);
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
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  double weights_sum = 0.0;
  for (auto& particle : particles) {
    // Get predicted landmarks within sensor range
    std::vector<LandmarkObs> predicted_landmarks;
    for (auto landmark : map_landmarks.landmark_list) {
      if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range) {
        LandmarkObs predicted_landmark;
        predicted_landmark.id = landmark.id_i;
        predicted_landmark.x = landmark.x_f;
        predicted_landmark.y = landmark.y_f;
        predicted_landmarks.push_back(predicted_landmark);
      }
    }

    std::vector<LandmarkObs> trans_obs;
    // To map coordinates
    for (auto observation : observations) {
      trans_obs.push_back(transformsCoordinates(particle, observation));
    }

    dataAssociation(predicted_landmarks, trans_obs);

    particle.weight = 1.0f;
    for (auto observation : trans_obs) {
      LandmarkObs associate_landmark;
      for (auto landmark : predicted_landmarks) {
        if (landmark.id == observation.id) {
          associate_landmark = landmark;
          break;
        }
      }
      particle.weight *= multivariateGaussianProb(observation, associate_landmark, std_landmark);
    }
    weights_sum += particle.weight;
  }

  for (int i=0; i<particles.size(); i++) {
    particles[i].weight /= weights_sum;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::discrete_distribution<> d(weights.begin(), weights.end());
  default_random_engine gen;
  std::vector<Particle> new_particles;
  for (int i=0; i<particles.size(); i++) {
    new_particles.push_back(particles[d(gen)]);
  }
  particles = new_particles;
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

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
