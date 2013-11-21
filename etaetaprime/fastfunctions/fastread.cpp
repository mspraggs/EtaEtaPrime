#include <vector>
#include <cstdlib>
#include <cstdio>

#include <boost/python.hpp>
#include <boost/python/list.hpp>

using namespace std;
namespace py = boost::python;

vector<vector<double> > readCSV(const char filename[512])
{
  FILE* file = fopen(filename, "r");
  if (file == (FILE*)NULL) {
    cout << "Can't open file" << endl;
    exit(3);
  }

  vector<vector<double> > data;
  data.reserve(10000);

  int counter = 0;
  while ((!feof(file))) {
    vector<double> row(4, 0.0);
    int datum0, datum1;
    long double datum2, datum3;
    fscanf(file, "%d%d%Le%Le", &datum0, &datum1, &datum2, &datum3);
    row[0] = (double) datum0;
    row[1] = (double) datum1;
    row[2] = datum2;
    row[3] = datum3;
    data.push_back(row);
  }
  fclose(file);

  return data;
}

py::list vectorToList(const vector<vector<double> >& input)
{
  py::list output;

  for (int i = 0; i < input.size(); ++i) {
    py::list list;
    for (int j = 0; j < 4; ++j)
      list.append(input[i][j]);
    output.append(list);
  }
  
  return output;
}

py::list csvToList(const char filename[512])
{
  vector<vector<double> > data = readCSV(filename);
  py::list output = vectorToList(data);
  return output;
}

BOOST_PYTHON_MODULE(fastread)
{
  // For some reason this function returns a list with an extra element equal
  // to the penultimate item in the list. For this reason it doesn't work
  // properly.
  py::def("read_csv", csvToList);
}
