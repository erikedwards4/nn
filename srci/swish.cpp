//Includes
#include "swish.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u};
const size_t I = 1u, O = 1u;
double beta;

//Description
string descr;
descr += "Activation function.\n";
descr += "Gets the Swish function [Ramachandran et al. 2017] of each element of X.\n";
descr += "For each element: y = x/(1+exp(-beta*x)).\n";
descr += "\n";
descr += "For beta=0, this is a scaled identity function: y = x/2.\n";
descr += "For beta=1, this is the same as the SiLU activation function.\n";
descr += "For beta->Inf, this tends towards a ReLU function.\n";
descr += "\n";
descr += "Use -b (--beta) to specify the beta parameter [default=1].\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ swish -b10 X -o Y \n";
descr += "$ swish -b10 X > Y \n";
descr += "$ cat X | swish -b10 > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_dbl    *a_b = arg_dbln("b","beta","<dbl>",0,1,"beta param [default=1.0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get beta
if (a_b->count==0) { beta = 1.0; }
else if (a_b->dval[0]<0.0) { cerr << progstr+": " << __LINE__ << errstr << "beta must be nonnegative" << endl; return 1; }
else { beta = a_b->dval[0]; }

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
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    if (codee::swish_inplace_s(X,i1.N(),float(beta)))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X;
}

//Finish
