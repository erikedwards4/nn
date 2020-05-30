//Includes
#include "softclip.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2};
const size_t I = 1, O = 1;
double p;

//Description
string descr;
descr += "Activation function.\n";
descr += "Gets softclip function [Klimek & Perelstein 2020] for each element of X.\n";
descr += "This is nearly linear in [0 1], 0 for x<0, and 1 for x>1.\n";
descr += "For each element: y = (1/p) * log([1+exp(p*x)]/[1+exp(p*(x-1))]).\n";
descr += "\n";
descr += "Use -p (--p) to specify the p parameter [default=50].\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ softclip X -o Y \n";
descr += "$ softclip X > Y \n";
descr += "$ cat X | softclip > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_dbl    *a_p = arg_dbln("p","p","<dbl>",0,1,"p param [default=50.0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get p
if (a_p->count==0) { p = 50.0; }
else if (a_p->dval[0]<=0.0) { cerr << progstr+": " << __LINE__ << errstr << "p param must be positive" << endl; return 1; }
else { p = a_p->dval[0]; }

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
    if (openn::softclip_inplace_s(X,int(i1.N()),float(p)))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X;
}

//Finish

