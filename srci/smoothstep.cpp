//Includes
#include "smoothstep.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2};
const size_t I = 1, O = 1;
int n;

//Description
string descr;
descr += "Activation function.\n";
descr += "Gets smoothstep function of each element of X.\n";
descr += "This implementation only allows n=0 and n=1.\n";
descr += "\n";
descr += "If p=0, then this is the clamp function:\n";
descr += "For each element: y = 0,  if x<0   \n";
descr += "                  y = x,  if 0<x<1 \n";
descr += "                  y = 1,  if x>1   \n";
descr += "\n";
descr += "If p=1, then this has a sigmoid shape.\n";
descr += "This is cubic Hermite interpolation after clamping.\n";
descr += "For each element: y = 0,             if x<0  \n";
descr += "                  y = 3*x^2 - 2*x^3, if 0<x<1\n";
descr += "                  y = 1,             if x>1  \n";
descr += "\n";
descr += "Use -n (--n) to specify the n param [default=0].\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ smoothstep X -n1 -o Y \n";
descr += "$ smoothstep X -n1 > Y \n";
descr += "$ cat X | smoothstep -n1 > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_int    *a_n = arg_intn("n","n","<uint>",0,1,"n param (0 or 1) [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get n
n = (a_n->count==0) ? 0 : a_n->ival[0];
if (n!=0 && n!=1) { cerr << progstr+": " << __LINE__ << errstr << "n param must be 0 or 1" << endl; return 1; }

//Checks
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) found to be empty" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;

//Other prep

//Process
if (i1.T==1)
{
    float *X;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    if (openn::smoothstep_inplace_s(X,int(i1.N()),n))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X;
}

//Finish

