//Includes
//#include <chrono>
#include "lstm_peephole.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u};
const size_t I = 4u, O = 1u;
size_t dim, N, T;

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
descr += "Input X has size NxT or TxN, where N is the number of neurons\n";
descr += "and T is the number of observations (e.g. time points).\n";
descr += "\n";
descr += "X has N time-series because it stacks the 4 driving inputs into one matrix: \n";
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
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,Ui,Uf,Uo)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = 0u; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = size_t(a_d->ival[0]); }
if (dim>1u) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

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
if (dim==0u && i1.R%4u) { cerr << progstr+": " << __LINE__ << errstr << "num rows X must be multiple of 4 for dim=0" << endl; return 1; }
if (dim==1u && i1.C%4u) { cerr << progstr+": " << __LINE__ << errstr << "num cols X must be multiple of 4 for dim=1" << endl; return 1; }
if (dim==0u && i1.R!=4u*i2.R) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2-4 (Ui, Uf, Uo) must have size NxN" << endl; return 1; }
if (dim==1u && i1.C!=4u*i2.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2-4 (Ui, Uf, Uo) must have size NxN" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = (dim==0u) ? i1.R/4u : i1.R;
o1.C = (dim==1u) ? i1.C/4u : i1.C;
o1.S = i1.S; o1.H = i1.H;

//Other prep
N = (dim==0u) ? o1.R : o1.C;
T = (dim==0u) ? o1.C : o1.R;

//Process
if (i1.T==1u)
{
    float *X, *Ui, *Uf, *Uo;//, *Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    try { Ui = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (Ui)" << endl; return 1; }
    try { Uf = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (Uf)" << endl; return 1; }
    try { Uo = new float[i4.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (Uo)" << endl; return 1; }
    //try { Y = new float[o1.N()]; }
    //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(Ui),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (Ui)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(Uf),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (Uf)" << endl; return 1; }
    try { ifs4.read(reinterpret_cast<char*>(Uo),i4.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (Uo)" << endl; return 1; }
    //auto tic = chrono::high_resolution_clock::now();
    //if (codee::lstm_peephole_s(Y,X,Ui,Uf,Uo,N,T,i1.iscolmajor(),dim))
    if (codee::lstm_peephole_inplace_s(X,Ui,Uf,Uo,N,T,i1.iscolmajor(),dim))
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
                try { cblas_scopy((int)N,&X[4u*t*N],1,&Y[t*N],1); }
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

//Finish
