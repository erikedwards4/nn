//Includes
#include "elu.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u};
const size_t I = 1u, O = 1u;
double alpha;

//Description
string descr;
descr += "Activation function.\n";
descr += "Gets exponential linear unit (ELU) [Clevert et al. 2015] of each element of X.\n";
descr += "For each element: y = alpha*(exp(x)-1),  if x<0.\n";
descr += "                  y = x,                 if x>=0.\n";
descr += "\n";
descr += "For alpha=0, this is the usual ReLU.\n";
descr += "\n";
descr += "Use -a (--alpha) to specify alpha [default=0.01].\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ elu X -o Y \n";
descr += "$ elu X > Y \n";
descr += "$ cat X | elu > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_dbl    *a_a = arg_dbln("a","alpha","<dbl>",0,1,"alpha param [default=1.0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get alpha
if (a_a->count==0) { alpha = 0.01; }
else if (a_a->dval[0]<0.0) { cerr << progstr+": " << __LINE__ << errstr << "alpha param must be nonnegative" << endl; return 1; }
else { alpha = a_a->dval[0]; }

//Checks
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) found to be empty" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;

//Other prep

//Process
if (i1.T==1u)
{
    float *X;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    if (codee::elu_inplace_s(X,i1.N(),float(alpha)))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X;
}

//Finish
