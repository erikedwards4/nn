//Includes
#include "hardtanh.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u};
const size_t I = 1u, O = 1u;
double a, b;

//Description
string descr;
descr += "Activation function.\n";
descr += "Gets hardtanh of each element of X.\n";
descr += "For each element: y = min,  if x<min. \n";
descr += "                  y = max,  if x>max. \n";
descr += "                  y = x,    otherwise.\n";
descr += "\n";
descr += "Use -a (--min) to specify the range min [default=-1].\n";
descr += "Use -b (--max) to specify the range max [default=1].\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ hardtanh X -o Y \n";
descr += "$ hardtanh -a-2 -b2 X > Y \n";
descr += "$ cat X | hardtanh -a-2.5 -b4.5 > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_dbl    *a_a = arg_dbln("a","min","<dbl>",0,1,"min param [default=-1.0]");
struct arg_dbl    *a_b = arg_dbln("b","max","<dbl>",0,1,"max param [default=1.0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get a
a = (a_a->count==0) ? -1.0 : a_a->dval[0];

//Get b
b = (a_b->count==0) ? 1.0 : a_b->dval[0];

//Checks
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) found to be empty" << endl; return 1; }
if (a>b) { cerr << progstr+": " << __LINE__ << errstr << "a must be <= b" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;

//Other prep

//Process
if (i1.T==1u)
{
    float *X;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    if (codee::hardtanh_inplace_s(X,i1.N(),float(a),float(b)))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X;
}

//Finish
