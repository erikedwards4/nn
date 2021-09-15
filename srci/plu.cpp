//Includes
#include "plu.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u};
const size_t I = 1u, O = 1u;
double a, c;

//Description
string descr;
descr += "Activation function.\n";
descr += "Gets Piecewise Linear Unit (PLU) [Nicolae 2018] for each element of X.\n";
descr += "For each element: y = max(a*(x+c)-c,min(a*(x-c)+c,x)).\n";
descr += "\n";
descr += "This equals x in range [-c c], and line with slope a otherwise.\n";
descr += "Thus, for a = 1 it is the identity function, and usually 0<a<1.\n";
descr += "\n";
descr += "Use -a (--a) to specify the a parameter [default=0.1].\n";
descr += "\n";
descr += "Use -c (--c) to specify alpha [default=1.0].\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ plu X -o Y \n";
descr += "$ plu X > Y \n";
descr += "$ cat X | plu > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_dbl    *a_a = arg_dbln("a","a","<dbl>",0,1,"a parameter [default=0.1]");
struct arg_dbl    *a_c = arg_dbln("c","c","<dbl>",0,1,"c parameter [default=1.0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get a
if (a_a->count==0) { a = 0.1; }
else if (a_a->dval[0]<0.0) { cerr << progstr+": " << __LINE__ << errstr << "a param must be nonnegative" << endl; return 1; }
else { a = a_a->dval[0]; }
if (a_a->dval[0]>1.0) { cerr << progstr+": " << __LINE__ << warstr << "a param is usually in [0 1]" << endl; return 1; }

//Get c
if (a_c->count==0) { c = 1.0; }
else if (a_c->dval[0]<0.0) { cerr << progstr+": " << __LINE__ << errstr << "c param must be nonnegative" << endl; return 1; }
else { c = a_c->dval[0]; }

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
    if (codee::plu_inplace_s(X,i1.N(),float(a),float(c)))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X;
}

//Finish
