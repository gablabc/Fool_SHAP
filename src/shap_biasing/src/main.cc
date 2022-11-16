#include <iostream>
#include <fstream>
#include <lemon/smart_graph.h>
#include <lemon/network_simplex.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>

using namespace std;
using namespace lemon;

// User must define feature and distance function
using MyFeature = vector<pair<int, double>>; // = sparse vector
struct MyDatum {
	double coeff;
	MyFeature feature;
};



double distance(MyFeature x, MyFeature y) {
	// Univariate data
	if (x.size() == 1 & y.size() == 1 & x[0].first == y[0].first) {
		return abs(x[0].second - y[0].second);
	}
	//Multivariate data
	auto square = [&](double t) { return t * t; };
	double score = 0;
	for (int i = 0, j = 0; i != x.size() || j != y.size(); ) {
		if (i == x.size()) {
			score += square(y[j].second);
			++j;
		} else if (j == y.size()) {
			score += square(x[i].second);
			++i;
		} else if (x[i].first < y[j].first) {
			score += square(x[i].second);
			++i;
		} else if (x[i].first > y[j].first) {
			score += square(y[j].second);
			++j;
		} else {
			score += square(x[i].second - y[j].second);
			++i;
			++j;
		} 
	}
	return sqrt(score);
}



vector<MyDatum> readFileData(string filename) {
	vector<MyDatum> data;
	ifstream ifs(filename);

	for (string line; getline(ifs, line); ) {
		if (line[0] == '#') continue;
		MyDatum datum;
		for (int i = 0; i < line.size(); ++i) 
			if (line[i] == ':') line[i] = ' ';
		stringstream ss(line);
		ss >> datum.coeff;
		int index;
		double value;
		while (ss >> index >> value) {
			datum.feature.push_back(make_pair(index, value));
		}
		sort(datum.feature.begin(), datum.feature.end());
		data.push_back(datum);
	}
	return data;
}



vector<double> readFileBounds(string filename){
    ifstream ifs(filename);
    vector<double> bounds;
    int N(0);
    for (string line; getline(ifs, line); ) {
        stringstream ss(line);
        double feature;
        while (ss >> feature){
            bounds.push_back(feature);
        }
        N++;
    }
    return bounds;
}



template <class Datum>
void MCF(vector<Datum> data, vector<double> bounds, double lambda) {
	double N = data.size();

	SmartDigraph g;
	SmartDigraph::ArcMap<double> capacity(g);
	SmartDigraph::ArcMap<double> cost(g);

	// supersource, superterminal
	SmartDigraph::Node s = g.addNode();
	SmartDigraph::Node t = g.addNode();

	// left vertices
	vector<SmartDigraph::Node> left;
	vector<SmartDigraph::Arc> incomming;
	for (int i = 0; i < data.size(); ++i) {
		left.push_back(g.addNode());
		SmartDigraph::Arc a = g.addArc(s, left[i]);
		capacity[a] = bounds[i];
		cost[a] = data[i].coeff;
		incomming.push_back(a);
	}

	// right vertices
	vector<SmartDigraph::Node> right;
	for (int i = 0; i < data.size(); ++i) {
		right.push_back(g.addNode());
		SmartDigraph::Arc a = g.addArc(right[i], t);
		capacity[a] = 1; 
		cost[a] = 0;
	}

	// pair-wise distances
	for (int i = 0; i < left.size(); ++i) {
		for (int j = 0; j < right.size(); ++j) {
			SmartDigraph::Arc a = g.addArc(left[i], right[j]);
			capacity[a] = 1;
			cost[a] = lambda * distance(data[i].feature, data[j].feature);
		}
	}

	NetworkSimplex<SmartDigraph, double, double> ns(g);
	ns.upperMap(capacity);
	ns.costMap(cost);
	ns.stSupply(s, t, N);

	bool res = ns.run();
	if (!res) cerr << "infeasible" << endl;
	SmartDigraph::ArcMap<double> flow(g);
	ns.flowMap(flow);

	// Print the solution
	vector<int> weight(N);
	for (int i = 0; i < N; ++i) {
		weight[i] = flow[incomming[i]];
		cout << weight[i] << endl;
	}
}



int main(int argc, char *argv[]) {
	// get the data
	vector<MyDatum> data  = readFileData(argv[1]);
	vector<double> bounds = readFileBounds(argv[2]);
	double lambda = stod(argv[3]);

	// run the MCF
	MCF(data, bounds, lambda);
}
