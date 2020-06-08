//Includes
#include "deadzone.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2};
const size_t I = 1, O = 1;
double delta;

//Description
string descr;
descr += "Does hard limiter with deadzone for each element of X.\n";
descr += "For each element: y = -1, if x<-d\n";
descr += "                  y =  0, if -d<=x<=d\n";
descr += "                  y =  1, if x>d\n";
descr += "\n";
descr += "Use -d (--delta) to specify the deadzone threshold [default=0].\n";
descr += "For d=0, this is the signum (sign) function.\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ deadzone X -d0.5 -o Y \n";
descr += "$ deadzone X -d0.5 > Y \n";
descr += "$ cat X | deadzone -d0.5 > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_dbl    *a_d = arg_dbln("d","delta","<dbl>",0,1,"deadzone delta [default=0.0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get delta
delta = (a_d->count==0) ? 0.0 : a_d->dval[0];

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
    if (openn::deadzone_inplace_s(X,int(i1.N()),float(delta)))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X;
}

//Finish

