//@author Erik Edwards
//@date 2018-present
//@license BSD 3-clause


#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string>
#include <cstring>
#include <valarray>
#include <unordered_map>
#include <argtable2.h>
#include "../util/cmli.hpp"
//#include <chrono>
#include <cblas.h>
#include "fukushima.c"

#ifdef I
#undef I
#endif


int main(int argc, char *argv[])
{
    using namespace std;


    //Declarations
    int ret = 0;
    const string errstr = ": \033[1;31merror:\033[0m ";
    const string warstr = ": \033[1;35mwarning:\033[0m ";
    const string progstr(__FILE__,string(__FILE__).find_last_of("/")+1,strlen(__FILE__)-string(__FILE__).find_last_of("/")-5);
    const valarray<size_t> oktypes = {1u,2u};
    const size_t I = 1u, O = 1u;
    ifstream ifs1; ofstream ofs1;
    int8_t stdi1, stdo1, wo1;
    ioinfo i1, o1;
    size_t dim, N, T;


    //Description
    string descr;
    descr += "Neural soma stage.\n";
    descr += "Does Fukushima model for driving inputs X.\n";
    descr += "\n";
    descr += "X has size 2NxT or Tx2N, where N is the number of neurons\n";
    descr += "and T is the number of observations (e.g. time points).\n";
    descr += "\n";
    descr += "Use -d (--dim) to specify the dimension (axis) of length 2N [default=0].\n";
    descr += "\n";
    descr += "This model divides the excitatory (E) part of X by the inhibitory (I) part.\n";
    descr += "as a quick model of shunting inhibition at the soma level.\n";
    descr += "Thus, the input X has 2 time-series per neuron, one for E and one for I.\n";
    descr += "By the original model, the input would come from wx, i.e. apply weights W\n";
    descr += "by matrix multiplication. In this case, W should also have E and I parts.\n";
    descr += "\n";
    descr += "For dim=0, Y[n,t] = (1+X[n,t])/(1+X[n+N,t]) - 1. \n";
    descr += "with sizes X: 2N x T \n";
    descr += "           Y:  N x T \n";
    descr += "\n";
    descr += "For dim=1, Y[t,n] = (1+X[t,n])/(1+X[t,n+N]) - 1. \n";
    descr += "with sizes X:  T x 2N \n";
    descr += "           Y:  T x N  \n";
    descr += "\n";
    descr += "X has 2N time-series because it stacks E and I parts into one matrix:\n";
    descr += "X = [Xe; Xi] for dim=0, and X = [Xe Xi] for dim=1. \n";
    descr += "\n";
    descr += "The output Y should be passed through an activation function with range [0 1].\n";
    descr += "The original model used a ReLU activation function, but that is not \n";
    descr += "included here so that other activation functions can be tried. \n";
    descr += "\n";
    descr += "Examples:\n";
    descr += "$ fukushima X -o Y \n";
    descr += "$ fukushima X > Y \n";
    descr += "$ cat X | fukushima > Y \n";


    //Argtable
    int nerrs;
    struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
    struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
    struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");
    struct arg_lit *a_help = arg_litn("h","help",0,1,"display this help and exit");
    struct arg_end  *a_end = arg_end(5);
    void *argtable[] = {a_fi, a_d, a_fo, a_help, a_end};
    if (arg_nullcheck(argtable)!=0) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating argtable" << endl; return 1; }
    nerrs = arg_parse(argc, argv, argtable);
    if (a_help->count>0)
    {
        cout << "Usage: " << progstr; arg_print_syntax(stdout, argtable, "\n");
        cout << endl; arg_print_glossary(stdout, argtable, "  %-25s %s\n");
        cout << endl << descr; return 1;
    }
    if (nerrs>0) { arg_print_errors(stderr,a_end,(progstr+": "+to_string(__LINE__)+errstr).c_str()); return 1; }


    //Check stdin
    stdi1 = (a_fi->count==0 || strlen(a_fi->filename[0])==0 || strcmp(a_fi->filename[0],"-")==0);
    if (stdi1>0 && isatty(fileno(stdin))) { cerr << progstr+": " << __LINE__ << errstr << "no stdin detected" << endl; return 1; }


    //Check stdout
    if (a_fo->count>0) { stdo1 = (strlen(a_fo->filename[0])==0 || strcmp(a_fo->filename[0],"-")==0); }
    else { stdo1 = (!isatty(fileno(stdout))); }
    wo1 = (stdo1 || a_fo->count>0);


    //Open input
    if (stdi1) { ifs1.copyfmt(cin); ifs1.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs1.open(a_fi->filename[0]); }
    if (!ifs1) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file" << endl; return 1; }


    //Read input header
    if (!read_input_header(ifs1,i1)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file" << endl; return 1; }
    if ((i1.T==oktypes).sum()==0)
    {
        cerr << progstr+": " << __LINE__ << errstr << "input data type must be in " << "{";
        for (auto o : oktypes) { cerr << int(o) << ((o==oktypes[oktypes.size()-1u]) ? "}" : ","); }
        cerr << endl; return 1;
    }


    //Get options

    //Get dim
    if (a_d->count==0) { dim = 0u; }
    else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
    else { dim = size_t(a_d->ival[0]); }
    if (dim>1u) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }


    //Checks
    if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) found to be empty" << endl; return 1; }
    if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) must be a matrix" << endl; return 1; }
    if (dim==0u && i1.R%2u) { cerr << progstr+": " << __LINE__ << errstr << "num rows X must be even for dim=0" << endl; return 1; }
    if (dim==1u && i1.C%2u) { cerr << progstr+": " << __LINE__ << errstr << "num cols X must be even for dim=1" << endl; return 1; }


    //Set output header info
    o1.F = i1.F; o1.T = i1.T;
    o1.R = (dim==0u) ? i1.R/2u : i1.R;
    o1.C = (dim==1u) ? i1.C/2u : i1.C;
    o1.S = i1.S; o1.H = i1.H;


    //Open output
    if (wo1)
    {
        if (stdo1) { ofs1.copyfmt(cout); ofs1.basic_ios<char>::rdbuf(cout.rdbuf()); } else { ofs1.open(a_fo->filename[0]); }
        if (!ofs1) { cerr << progstr+": " << __LINE__ << errstr << "problem opening output file 1" << endl; return 1; }
    }


    //Write output header
    if (wo1 && !write_output_header(ofs1,o1)) { cerr << progstr+": " << __LINE__ << errstr << "problem writing header for output file 1" << endl; return 1; }


    //Other prep
    N = (dim==0u) ? o1.R : o1.C;
    T = (dim==0u) ? o1.C : o1.R;
    

    //Process
    if (i1.T==1u)
    {
        float *X; //*Y;
        try { X = new float[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
        //try { Y = new float[o1.N()]; }
        //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
        //auto tic = chrono::high_resolution_clock::now();
        //if (codee::fukushima_s(Y,X,N,T,i1.iscolmajor(),dim))
        if (codee::fukushima_inplace_s(X,N,T,i1.iscolmajor(),dim))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
        if (wo1)
        {
            //try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
            if ((dim==0u && o1.isrowmajor()) || (dim==1u && o1.iscolmajor()))
            {
                try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
                catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
            }
            else
            {
                float *Y;
                try { Y = new float[o1.N()]; }
                catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
                for (size_t t=0u; t<T; ++t)
                {
                    try { cblas_scopy((int)N,&X[2u*t*N],1,&Y[t*N],1); }
                    catch(...) { cerr << progstr+": " << __LINE__ << errstr << "problem copying to output file (Y)" << endl; return 1; }
                }
                try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
                catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
                delete[] Y;
            }
        }
        delete[] X; //delete[] Y;
        //auto toc = chrono::high_resolution_clock::now();
        //auto dur = chrono::duration_cast<chrono::microseconds>(toc-tic);
        //cerr << dur.count()/1000.0 << " ms" << endl; 
    }
    else if (i1.T==2)
    {
        double *X; //*Y;
        try { X = new double[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
        //try { Y = new double[o1.N()]; }
        //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
        //auto tic = chrono::high_resolution_clock::now();
        //if (codee::fukushima_d(Y,X,N,T,i1.iscolmajor(),dim))
        if (codee::fukushima_inplace_d(X,N,T,i1.iscolmajor(),dim))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
        if (wo1)
        {
            //try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
            if ((dim==0u && o1.isrowmajor()) || (dim==1u && o1.iscolmajor()))
            {
                try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
                catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
            }
            else
            {
                double *Y;
                try { Y = new double[o1.N()]; }
                catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
                for (size_t t=0u; t<T; ++t)
                {
                    try { cblas_dcopy((int)N,&X[2u*t*N],1,&Y[t*N],1); }
                    catch(...) { cerr << progstr+": " << __LINE__ << errstr << "problem copying to output file (Y)" << endl; return 1; }
                }
                try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
                catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
                delete[] Y;
            }
        }
        delete[] X; //delete[] Y;
        //auto toc = chrono::high_resolution_clock::now();
        //auto dur = chrono::duration_cast<chrono::microseconds>(toc-tic);
        //cerr << dur.count()/1000.0 << " ms" << endl; 
    }
    else
    {
        cerr << progstr+": " << __LINE__ << errstr << "data type not supported" << endl; return 1;
    }
    

    //Exit
    return ret;
}

