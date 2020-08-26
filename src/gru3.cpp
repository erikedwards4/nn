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
#include "gru3.c"

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
    const size_t I = 6u, O = 1u;
    ifstream ifs1, ifs2, ifs3, ifs4, ifs5, ifs6; ofstream ofs1;
    int8_t stdi1, stdi2, stdi3, stdi4, stdi5, stdi6, stdo1, wo1;
    ioinfo i1, i2, i3, i4, i5, i6, o1;
    size_t dim, N, T;


    //Description
    string descr;
    descr += "Neural soma stage.\n";
    descr += "Does GRU (gated recurrent unit) model for driving inputs X, Xr, Xz.\n";
    descr += "This is the fully gated unit with 3 inputs (see also gru_min2).\n";
    descr += "\n";
    descr += "X is the usual input and Xr, Xz are the inputs for the reset and update gates.\n";
    descr += "All have size NxT or TxN, where N is the number of neurons,\n";
    descr += "and T is the number of observations (e.g. time points).\n";
    descr += "\n";
    descr += "Use -d (--dim) to specify the dimension (axis) of length N [default=0].\n";
    descr += "\n";
    descr += "For dim=0, R[:,t] = sig{Xr[:,t] + Ur*Y[:,t-1]} \n";
    descr += "           Z[:,t] = sig{Xz[:,t] + Uz*Y[:,t-1]} \n";
    descr += "           H[:,t] = R[:,t].*Y[:,t-1] \n";
    descr += "           Y[:,t] = Z[:,t].*Y[:,t-1] + (1-Z[:,t]).*tanh{X[:,t] + U*H[:,t]} \n";
    descr += "with sizes X, Xr, Xz: N x T \n";
    descr += "           U, Ur, Uz: N x N \n";
    descr += "           Y        : N x T \n";
    descr += "\n";
    descr += "For dim=1, R[t,:] = sig{Xr[t,:] + Y[t-1,:]*Ur} \n";
    descr += "           Z[t,:] = sig{Xz[t,:] + Y[t-1,:]*Uz} \n";
    descr += "           H[t,:] = R[t,:].*Y[t-1,:] \n";
    descr += "           Y[t,:] = Z[t,:].*Y[t-1,:] + (1-Z[t,:]).*tanh{X[t,:] + H[t,:]*U} \n";
    descr += "with sizes X, Xr, Xz: T x N \n";
    descr += "           U, Ur, Uz: N x N \n";
    descr += "           Y        : T x N \n";
    descr += "\n";
    descr += "Examples:\n";
    descr += "$ gru3 X Xr Xz U Ur Uz -o Y \n";
    descr += "$ gru3 X Xr Xz U Ur Uz > Y \n";
    descr += "$ cat X | gru3 - Xr Xz U Ur Uz > Y \n";


    //Argtable
    int nerrs;
    struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,Xr,Xz,U,Ur,Uz)");
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
    stdi2 = (a_fi->count<=1 || strlen(a_fi->filename[1])==0 || strcmp(a_fi->filename[1],"-")==0);
    stdi3 = (a_fi->count<=2 || strlen(a_fi->filename[2])==0 || strcmp(a_fi->filename[2],"-")==0);
    stdi4 = (a_fi->count<=3 || strlen(a_fi->filename[3])==0 || strcmp(a_fi->filename[3],"-")==0);
    stdi5 = (a_fi->count<=4 || strlen(a_fi->filename[4])==0 || strcmp(a_fi->filename[4],"-")==0);
    stdi6 = (a_fi->count<=5 || strlen(a_fi->filename[5])==0 || strcmp(a_fi->filename[5],"-")==0);
    if (stdi1+stdi2+stdi3+stdi4+stdi5+stdi6>1) { cerr << progstr+": " << __LINE__ << errstr << "can only use stdin for one input" << endl; return 1; }
    if (stdi1+stdi2+stdi3+stdi4+stdi5+stdi6>0 && isatty(fileno(stdin))) { cerr << progstr+": " << __LINE__ << errstr << "no stdin detected" << endl; return 1; }


    //Check stdout
    if (a_fo->count>0) { stdo1 = (strlen(a_fo->filename[0])==0 || strcmp(a_fo->filename[0],"-")==0); }
    else { stdo1 = (!isatty(fileno(stdout))); }
    wo1 = (stdo1 || a_fo->count>0);


    //Open inputs
    if (stdi1) { ifs1.copyfmt(cin); ifs1.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs1.open(a_fi->filename[0]); }
    if (!ifs1) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 1" << endl; return 1; }
    if (stdi2) { ifs2.copyfmt(cin); ifs2.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs2.open(a_fi->filename[1]); }
    if (!ifs2) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 2" << endl; return 1; }
    if (stdi3) { ifs3.copyfmt(cin); ifs3.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs3.open(a_fi->filename[2]); }
    if (!ifs3) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 3" << endl; return 1; }
    if (stdi4) { ifs4.copyfmt(cin); ifs4.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs4.open(a_fi->filename[3]); }
    if (!ifs4) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 4" << endl; return 1; }
    if (stdi5) { ifs5.copyfmt(cin); ifs5.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs5.open(a_fi->filename[4]); }
    if (!ifs5) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 5" << endl; return 1; }
    if (stdi6) { ifs6.copyfmt(cin); ifs6.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs6.open(a_fi->filename[5]); }
    if (!ifs6) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 6" << endl; return 1; }


    //Read input headers
    if (!read_input_header(ifs1,i1)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 1" << endl; return 1; }
    if (!read_input_header(ifs2,i2)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 2" << endl; return 1; }
    if (!read_input_header(ifs3,i3)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 3" << endl; return 1; }
    if (!read_input_header(ifs4,i4)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 4" << endl; return 1; }
    if (!read_input_header(ifs5,i5)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 5" << endl; return 1; }
    if (!read_input_header(ifs6,i6)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 6" << endl; return 1; }
    if ((i1.T==oktypes).sum()==0 || (i2.T==oktypes).sum()==0 || (i3.T==oktypes).sum()==0 || (i4.T==oktypes).sum()==0 || (i5.T==oktypes).sum()==0 || (i6.T==oktypes).sum()==0)
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
    if (i1.T!=i2.T || i1.T!=i3.T || i1.T!=i4.T || i1.T!=i5.T || i1.T!=i6.T)
    { cerr << progstr+": " << __LINE__ << errstr << "all inputs must have the same data type" << endl; return 1; }
    if (!i1.isvec() && (i1.iscolmajor()!=i2.iscolmajor() || i1.iscolmajor()!=i3.iscolmajor() || i1.iscolmajor()!=i4.iscolmajor() || i1.iscolmajor()!=i5.iscolmajor() || i1.iscolmajor()!=i6.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "all inputs must have the same row/col major format" << endl; return 1; }
    if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
    if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Xr) found to be empty" << endl; return 1; }
    if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (Xz) found to be empty" << endl; return 1; }
    if (i4.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (U) found to be empty" << endl; return 1; }
    if (i5.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (Ur) found to be empty" << endl; return 1; }
    if (i6.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 6 (Uz) found to be empty" << endl; return 1; }
    if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) must be a matrix" << endl; return 1; }
    if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Xr) must be a matrix" << endl; return 1; }
    if (!i3.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (Xz) must be a matrix" << endl; return 1; }
    if (!i4.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (U) must be a matrix" << endl; return 1; }
    if (!i5.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (Ur) must be a matrix" << endl; return 1; }
    if (!i6.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 6 (Uz) must be a matrix" << endl; return 1; }
    if (i1.R!=i2.R || i1.C!=i2.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 1-3 (X, Xr, Xz) must have the same size" << endl; return 1; }
    if (i1.R!=i3.R || i1.C!=i3.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 1-3 (X, Xr, Xz) must have the same size" << endl; return 1; }
    if (i4.R!=i5.R || i4.C!=i5.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 4-6 (U, Ur, Uz) must have the same size" << endl; return 1; }
    if (i4.R!=i6.R || i4.C!=i6.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 4-6 (U, Ur, Uz) must have the same size" << endl; return 1; }
    if (i4.R!=i4.C || i5.R!=i5.C || i6.R!=i6.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 4-6 (U, Ur, Uz) must be square" << endl; return 1; }
    if (dim==0u && i1.R!=i4.R) { cerr << progstr+": " << __LINE__ << errstr << "inputs 4-6 (U, Ur, Uz) must have size NxN" << endl; return 1; }
    if (dim==1u && i1.C!=i4.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 4-6 (U, Ur, Uz) must have size NxN" << endl; return 1; }


    //Set output header info
    o1.F = i1.F; o1.T = i1.T;
    o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;


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
        float *X, *Xr, *Xz, *U, *Ur, *Uz, *Y;
        try { X = new float[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
        try { Xr = new float[i2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (Xr)" << endl; return 1; }
        try { Xz = new float[i3.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (Xz)" << endl; return 1; }
        try { U = new float[i4.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (U)" << endl; return 1; }
        try { Ur = new float[i5.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 5 (Ur)" << endl; return 1; }
        try { Uz = new float[i6.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 6 (Uz)" << endl; return 1; }
        try { Y = new float[o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
        try { ifs2.read(reinterpret_cast<char*>(Xr),i2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (Xr)" << endl; return 1; }
        try { ifs3.read(reinterpret_cast<char*>(Xz),i3.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (Xz)" << endl; return 1; }
        try { ifs4.read(reinterpret_cast<char*>(U),i4.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (U)" << endl; return 1; }
        try { ifs5.read(reinterpret_cast<char*>(Ur),i5.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 5 (Ur)" << endl; return 1; }
        try { ifs6.read(reinterpret_cast<char*>(Uz),i6.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 6 (Uz)" << endl; return 1; }
        //auto tic = chrono::high_resolution_clock::now();
        if (codee::gru3_s(Y,X,Xr,Xz,U,Ur,Uz,N,T,i1.iscolmajor(),dim))
        //if (codee::gru3_inplace_s(X,Xr,Xz,U,Ur,Uz,N,T,i1.iscolmajor(),dim))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] Xr; delete[] Xz; delete[] U; delete[] Ur; delete[] Uz; delete[] Y;
        //auto toc = chrono::high_resolution_clock::now();
        //auto dur = chrono::duration_cast<chrono::microseconds>(toc-tic);
        //cerr << dur.count()/1000.0 << " ms" << endl; 
    }
    else if (i1.T==2)
    {
        double *X, *Xr, *Xz, *U, *Ur, *Uz, *Y;
        try { X = new double[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
        try { Xr = new double[i2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (Xr)" << endl; return 1; }
        try { Xz = new double[i3.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (Xz)" << endl; return 1; }
        try { U = new double[i4.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (U)" << endl; return 1; }
        try { Ur = new double[i5.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 5 (Ur)" << endl; return 1; }
        try { Uz = new double[i6.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 6 (Uz)" << endl; return 1; }
        try { Y = new double[o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
        try { ifs2.read(reinterpret_cast<char*>(Xr),i2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (Xr)" << endl; return 1; }
        try { ifs3.read(reinterpret_cast<char*>(Xz),i3.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (Xz)" << endl; return 1; }
        try { ifs4.read(reinterpret_cast<char*>(U),i4.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (U)" << endl; return 1; }
        try { ifs5.read(reinterpret_cast<char*>(Ur),i5.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 5 (Ur)" << endl; return 1; }
        try { ifs6.read(reinterpret_cast<char*>(Uz),i6.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 6 (Uz)" << endl; return 1; }
        //auto tic = chrono::high_resolution_clock::now();
        if (codee::gru3_d(Y,X,Xr,Xz,U,Ur,Uz,N,T,i1.iscolmajor(),dim))
        //if (codee::gru3_inplace_d(X,Xr,Xz,U,Ur,Uz,N,T,i1.iscolmajor(),dim))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] Xr; delete[] Xz; delete[] U; delete[] Ur; delete[] Uz; delete[] Y;
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

