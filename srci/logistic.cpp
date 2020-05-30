//Includes
#include "logistic.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2};
const size_t I = 1, O = 1;
double alpha;

//Description
string descr;
descr += "Activation function.\n";
descr += "Gets logistic function (basic sigmoid) of each element of X.\n";
descr += "For each element: y = 1/(1+exp(-x)).\n";
descr += "\n";
descr += "A generalized logistic function can also be given by the alpha parameter.\n";
descr += "For each element: y = 1/(1+exp(-x))^alpha.\n";
descr += "\n";
descr += "Use -a (--alpha) to specify alpha [default=1].\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ logistic X -o Y \n";
descr += "$ logistic X > Y \n";
descr += "$ cat X | logistic > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_dbl    *a_a = arg_dbln("a","alpha","<dbl>",0,1,"alpha param [default=1.0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get alpha
if (a_a->count==0) { alpha = 1.0; }
else if (a_a->dval[0]<=0.0) { cerr << progstr+": " << __LINE__ << errstr << "alpha param must be positive" << endl; return 1; }
else { alpha = a_a->dval[0]; }

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
    if (openn::logistic_inplace_s(X,int(i1.N()),float(alpha)))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X;
}

//Finish

