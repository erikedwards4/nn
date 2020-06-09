//@author Erik Edwards
//@date 2019-2020


#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string>
#include <cstring>
#include <valarray>
#include <complex>
#include <unordered_map>
#include <argtable2.h>
#include "/home/erik/codee/cmli/cmli.hpp"
//#include <chrono>
#include "lstm_peephole.c"

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
    const valarray<uint8_t> oktypes = {1,2};
    const size_t I = 4, O = 1;
    ifstream ifs1, ifs2, ifs3, ifs4; ofstream ofs1;
    int8_t stdi1, stdi2, stdi3, stdi4, stdo1, wo1;
    ioinfo i1, i2, i3, i4, o1;
    int dim, N, T;


    //Description
    string descr;
    descr += "Neural soma stage.\n";
    descr += "Does peephole LSTM (long short-term memory) model for driving \n";
    descr += "inputs Xc, Xi, Xf, Xo, where Xc is the input to the cell, and\n";
    descr += "Xi, Xf, Xo are the inputs for the input, forget, output gates.\n";
    descr += "\n";
    descr += "All have size NxT or TxN, where N is the number of neurons,\n";
    descr += "and T is the number of observations (e.g. time points).\n";
    descr += "\n";
    descr += "Input X has size 4NxT or Tx4N, where N is the number of neurons\n";
    descr += "and T is the number of observations (e.g. time points).\n";
    descr += "\n";
    descr += "X has 4N time-series because it stacks the 4 driving inputs into one matrix: \n";
    descr += "For dim==0, X = [Xc; Xi; Xf; Xo], and for dim==1, X = [Xc Xi Xf Xo]. \n";
    descr += "\n";
    descr += "Use -d (--dim) to specify the dimension (axis) of length N [default=0].\n";
    descr += "\n";
    descr += "For dim=0, I[:,t] = sig{Xi[:,t] + Ui*C[:,t-1]} \n";
    descr += "           F[:,t] = sig{Xf[:,t] + Uf*C[:,t-1]} \n";
    descr += "           O[:,t] = sig{Xo[:,t] + Uo*C[:,t-1]} \n";
    descr += "           C[:,t] = F[:,t].*C[:,t-1] + I[:,t].*sig{Xc[t,:]} \n";
    descr += "           Y[:,t] = tanh{O[:,t].*C[:,t]} \n";
    descr += "with sizes Xc, Xi, Xf, Xo: N x T \n";
    descr += "               Ui, Uf, Uo: N x N \n";
    descr += "                        Y: N x T \n";
    descr += "\n";
    descr += "For dim=1, I[t,:] = sig{Xi[t,:] + C[t-1,:]*Ui} \n";
    descr += "           F[t,:] = sig{Xf[t,:] + C[t-1,:]*Uf} \n";
    descr += "           O[t,:] = sig{Xo[t,:] + C[t-1,:]*Uo} \n";
    descr += "           C[t,:] = F[t,:].*C[t-1,:] + I[t,:].*sig{Xc[t,:]} \n";
    descr += "           Y[t,:] = tanh{O[t,:].*C[t,:]} \n";
    descr += "with sizes Xc, Xi, Xf, Xo: T x N \n";
    descr += "               Ui, Uf, Uo: N x N \n";
    descr += "                        Y: T x N \n";
    descr += "\n";
    descr += "However, here the final tanh nonlinearity is not included,\n";
    descr += "so that other activation or output functions can be used.\n";
    descr += "\n";
    descr += "Examples:\n";
    descr += "$ lstm_peephole X Uc Ui Uf Uo -o Y \n";
    descr += "$ lstm_peephole X Uc Ui Uf Uo > Y \n";
    descr += "$ cat X | lstm_peephole - Uc Ui Uf Uo | tanh > Y \n";


    //Argtable
    int nerrs;
    struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,Ui,Uf,Uo)");
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
    if (stdi1+stdi2+stdi3+stdi4>1) { cerr << progstr+": " << __LINE__ << errstr << "can only use stdin for one input" << endl; return 1; }
    if (stdi1+stdi2+stdi3+stdi4>0 && isatty(fileno(stdin))) { cerr << progstr+": " << __LINE__ << errstr << "no stdin detected" << endl; return 1; }


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


    //Read input headers
    if (!read_input_header(ifs1,i1)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 1" << endl; return 1; }
    if (!read_input_header(ifs2,i2)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 2" << endl; return 1; }
    if (!read_input_header(ifs3,i3)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 3" << endl; return 1; }
    if (!read_input_header(ifs4,i4)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 4" << endl; return 1; }
    if ((i1.T==oktypes).sum()==0 || (i2.T==oktypes).sum()==0 || (i3.T==oktypes).sum()==0 || (i4.T==oktypes).sum()==0)
    {
        cerr << progstr+": " << __LINE__ << errstr << "input data type must be in " << "{";
        for (auto o : oktypes) { cerr << int(o) << ((o==oktypes[oktypes.size()-1]) ? "}" : ","); }
        cerr << endl; return 1;
    }


    //Get options

    //Get dim
    if (a_d->count==0) { dim = 0; }
    else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
    else { dim = a_d->ival[0]; }
    if (dim>1) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }


    //Checks
    if (i1.T!=i2.T || i1.T!=i3.T || i1.T!=i4.T)
    { cerr << progstr+": " << __LINE__ << errstr << "all inputs must have the same data type" << endl; return 1; }
    if (!i1.isvec() && (i1.iscolmajor()!=i2.iscolmajor() || i1.iscolmajor()!=i3.iscolmajor() || i1.iscolmajor()!=i4.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "all inputs must have the same row/col major format" << endl; return 1; }
    if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
    if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Ui) found to be empty" << endl; return 1; }
    if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (Uf) found to be empty" << endl; return 1; }
    if (i4.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (Uo) found to be empty" << endl; return 1; }
    if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) must be a matrix" << endl; return 1; }
    if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Ui) must be a matrix" << endl; return 1; }
    if (!i3.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (Uf) must be a matrix" << endl; return 1; }
    if (!i4.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (Uo) must be a matrix" << endl; return 1; }
    if (i2.R!=i3.R || i2.C!=i3.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2-4 (Ui, Uf, Uo) must have the same size" << endl; return 1; }
    if (i2.R!=i4.R || i2.C!=i4.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2-4 (Ui, Uf, Uo) must have the same size" << endl; return 1; }
    if (dim==0 && i1.R%4u) { cerr << progstr+": " << __LINE__ << errstr << "num rows X must be multiple of 4 for dim=0" << endl; return 1; }
    if (dim==1 && i1.C%4u) { cerr << progstr+": " << __LINE__ << errstr << "num cols X must be multiple of 4 for dim=1" << endl; return 1; }
    if (dim==0 && i1.R!=4u*i2.R) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2-4 (Ui, Uf, Uo) must have size NxN" << endl; return 1; }
    if (dim==1 && i1.C!=4u*i2.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2-4 (Ui, Uf, Uo) must have size NxN" << endl; return 1; }


    //Set output header info
    o1.F = i1.F; o1.T = i1.T;
    o1.R = (dim==0) ? i1.R/4u : i1.R;
    o1.C = (dim==1) ? i1.C/4u : i1.C;
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
    N = (dim==0) ? int(o1.R) : int(o1.C);
    T = (dim==0) ? int(o1.C) : int(o1.R);
    

    //Process
    if (i1.T==1)
    {
        float *X, *Ui, *Uf, *Uo;//, *Y;
        try { X = new float[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
        try { Ui = new float[i2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (Ui)" << endl; return 1; }
        try { Uf = new float[i3.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (Uf)" << endl; return 1; }
        try { Uo = new float[i4.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (Uo)" << endl; return 1; }
        //try { Y = new float[o1.N()]; }
        //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
        try { ifs2.read(reinterpret_cast<char*>(Ui),i2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (Ui)" << endl; return 1; }
        try { ifs3.read(reinterpret_cast<char*>(Uf),i3.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (Uf)" << endl; return 1; }
        try { ifs4.read(reinterpret_cast<char*>(Uo),i4.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (Uo)" << endl; return 1; }
        //auto tic = chrono::high_resolution_clock::now();
        //if (openn::lstm_peephole_s(Y,X,Ui,Uf,Uo,N,T,dim,i1.iscolmajor()))
        if (openn::lstm_peephole_inplace_s(X,Ui,Uf,Uo,N,T,dim,i1.iscolmajor()))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
        if (wo1)
        {
            //try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
            if ((dim==0 && o1.isrowmajor()) || (dim==1 && o1.iscolmajor()))
            {
                try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
                catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
            }
            else
            {
                float *Y;
                try { Y = new float[o1.N()]; }
                catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
                for (int t=0; t<T; t++)
                {
                    try { cblas_scopy(N,&X[4*t*N],1,&Y[t*N],1); }
                    catch(...) { cerr << progstr+": " << __LINE__ << errstr << "problem copying to output file (Y)" << endl; return 1; }
                }
                try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
                catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
                delete[] Y;
            }
        }
        delete[] X; delete[] Ui; delete[] Uf; delete[] Uo; //delete[] Y;
        //auto toc = chrono::high_resolution_clock::now();
        //auto dur = chrono::duration_cast<chrono::microseconds>(toc-tic);
        //cerr << dur.count()/1000.0 << " ms" << endl; 
    }
    else if (i1.T==2)
    {
        double *X, *Ui, *Uf, *Uo;//, *Y;
        try { X = new double[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
        try { Ui = new double[i2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (Ui)" << endl; return 1; }
        try { Uf = new double[i3.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (Uf)" << endl; return 1; }
        try { Uo = new double[i4.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (Uo)" << endl; return 1; }
        //try { Y = new double[o1.N()]; }
        //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
        try { ifs2.read(reinterpret_cast<char*>(Ui),i2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (Ui)" << endl; return 1; }
        try { ifs3.read(reinterpret_cast<char*>(Uf),i3.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (Uf)" << endl; return 1; }
        try { ifs4.read(reinterpret_cast<char*>(Uo),i4.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (Uo)" << endl; return 1; }
        //auto tic = chrono::high_resolution_clock::now();
        //if (openn::lstm_peephole_d(Y,X,Ui,Uf,Uo,N,T,dim,i1.iscolmajor()))
        if (openn::lstm_peephole_inplace_d(X,Ui,Uf,Uo,N,T,dim,i1.iscolmajor()))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
        if (wo1)
        {
            //try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
            if ((dim==0 && o1.isrowmajor()) || (dim==1 && o1.iscolmajor()))
            {
                try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
                catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
            }
            else
            {
                double *Y;
                try { Y = new double[o1.N()]; }
                catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
                for (int t=0; t<T; t++)
                {
                    try { cblas_dcopy(N,&X[4*t*N],1,&Y[t*N],1); }
                    catch(...) { cerr << progstr+": " << __LINE__ << errstr << "problem copying to output file (Y)" << endl; return 1; }
                }
                try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
                catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
                delete[] Y;
            }
        }
        delete[] X; delete[] Ui; delete[] Uf; delete[] Uo; //delete[] Y;
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

