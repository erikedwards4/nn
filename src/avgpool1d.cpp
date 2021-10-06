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
#include "avgpool1d.c"

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
    const valarray<size_t> oktypes = {1u,2u,101u,102u};
    const size_t I = 1u, O = 1u;
    ifstream ifs1; ofstream ofs1;
    int8_t stdi1, stdo1, wo1;
    ioinfo i1, o1;
    size_t N, Li, Lo, Lk, str, dil;
    int pad, Ti, Tk, ceil_mode, pm;
    string pad_mode;


    //Description
    string descr;
    descr += "1d average-pooling, with full opts of conv1d (e.g., dilation).\n";
    descr += "\n";
    descr += "This is an input (IN) component for a layer of neurons.\n";
    descr += "There are N neurons in the layer,\n";
    descr += "and the layer gets inputs from N neurons.\n";
    descr += "\n";
    descr += "X has size N x Li for col-major.\n";
    descr += "X has size Li x N for row-major.\n";
    descr += "where Li is the input length (usually the number of time points).\n";
    descr += "\n";
    descr += "Lk is the kernel length (kernel_size, or time width).\n";
    descr += "There is no explicit kernel; this is by analogy to conv1.\n";
    descr += "Thus, Lk is the num samps included in each average.\n";
    descr += "\n";
    descr += "Y has size N x Lo for col-major.\n";
    descr += "Y has size Lo x N for row-major.\n";
    descr += "where:\n";
    descr += "Lo =  ceil[1 + (Li + 2*pad - dil*(Lk-1) - 1)/stride], for ceil_mode true.\n";
    descr += "Lo = floor[1 + (Li + 2*pad - dil*(Lk-1) - 1)/stride], for ceil_mode false.\n";
    descr += "\n";
    descr += "Include -c (--ceil_mode) to use ceil for Lo calculation [default=false].\n";
    descr += "\n";
    descr += "Use -s (--stride) to give the stride (step-size) in samples [default=1].\n";
    descr += "\n";
    descr += "Use -i (--dilation) to give the dilation factor [default=1].\n";
    descr += "\n";
    descr += "Use -p (--padding) to give the padding length in samples [default=0]\n";
    descr += "\n";
    descr += "Use -m (--pad_mode) to give the padding mode as [default='zeros']\n";
    descr += "The pad_mode can be 'zeros', 'reflect', 'repeat' or 'circular'.\n";
    descr += "The pad_mode can also be entered as 'z', 'ref', 'rep' or 'c'.\n";
    descr += "Use pad_mode 'no_count_pad' or 'n' to emulate count_include_pad=False.\n";
    descr += "\n";
    descr += "Examples:\n";
    descr += "$ avgpool1d -k5 X -o Y \n";
    descr += "$ avgpool1d -k9 -s2 -p10 -m'reflect' X > Y \n";
    descr += "$ cat X | avgpool1d -k12 -i2 -c -s5 -p10 -m'c' - > Y \n";


    //Argtable
    int nerrs;
    struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
    struct arg_int   *a_lk = arg_intn("k","kernel_size","<uint>",0,1,"kernel length in samps [default=1]");
    struct arg_int  *a_str = arg_intn("s","step","<uint>",0,1,"step size in samps [default=1]");
    struct arg_int  *a_dil = arg_intn("i","dilation","<uint>",0,1,"dilation factor [default=1]");
    struct arg_int  *a_pad = arg_intn("p","padding","<int>",0,1,"padding [default=0]");
    struct arg_str   *a_pm = arg_strn("m","pad_mode","<str>",0,1,"padding mode [default='zeros']");
    struct arg_lit   *a_cm = arg_litn("c","ceil_mode",0,1,"include to use ceil_mode [default=false]");
    struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");
    struct arg_lit *a_help = arg_litn("h","help",0,1,"display this help and exit");
    struct arg_end  *a_end = arg_end(5);
    void *argtable[] = {a_fi, a_lk, a_str, a_dil, a_pad, a_pm, a_cm, a_fo, a_help, a_end};
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
    stdi1 = (a_fi->count==0 || strlen(a_fi->filename[0])==0u || strcmp(a_fi->filename[0],"-")==0);
    if (stdi1>0 && isatty(fileno(stdin))) { cerr << progstr+": " << __LINE__ << errstr << "no stdin detected" << endl; return 1; }


    //Check stdout
    if (a_fo->count>0) { stdo1 = (strlen(a_fo->filename[0])==0u || strcmp(a_fo->filename[0],"-")==0); }
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

    //Get Lk
    if (a_lk->count==0) { Lk = 1u; }
    else if (a_lk->ival[0]<1) { cerr << progstr+": " << __LINE__ << errstr << "kernel_size must be positive" << endl; return 1; }
    else { Lk = size_t(a_lk->ival[0]); }

    //Get stride
    if (a_str->count==0) { str = 1u; }
    else if (a_str->ival[0]<1) { cerr << progstr+": " << __LINE__ << errstr << "stride must be positive" << endl; return 1; }
    else { str = size_t(a_str->ival[0]); }

    //Get dil
    if (a_dil->count==0) { dil = 1u; }
    else if (a_dil->ival[0]<1) { cerr << progstr+": " << __LINE__ << errstr << "dilation must be positive" << endl; return 1; }
    else { dil = size_t(a_dil->ival[0]); }

    //Get ceil_mode
    ceil_mode = (a_cm->count>0);

    //Get padding
    if (a_pad->count==0) { pad = 0; }
    else { pad = a_pad->ival[0]; }

    //Get pad_mode and pm
    if (a_pm->count==0) { pad_mode = "zeros"; }
    else
    {
    	try { pad_mode = string(a_pm->sval[0]); }
    	catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem getting string for pad_mode" << endl; return 1; }
    }
    for (string::size_type c=0u; c<pad_mode.size(); ++c) { pad_mode[c] = char(tolower(pad_mode[c])); }
    if (pad_mode=="z" || pad_mode=="zeros") { pm = 0; }
    else if (pad_mode=="rep" || pad_mode=="replicate") { pm = 1; }
    else if (pad_mode=="ref" || pad_mode=="reflect") { pm = 2; }
    else if (pad_mode=="c" || pad_mode=="circular") { pm = 3; }
    else if (pad_mode=="n" || pad_mode=="no_count_pad") { pm = 4; }
    else
    {
        cerr << progstr+": " << __LINE__ << errstr << "pad_mode must be in {'z','rep','ref','c','n'} or {'zeros','replicate','reflect','circular','no_count_pad'}" << endl;
        return 1;
    }


    //Checks
    N = i1.iscolmajor() ? i1.R : i1.C;
    Li = i1.iscolmajor() ? i1.C : i1.R;
    Ti = (int)Li + 2*pad;
    Tk = (int)(dil*(Lk-1u)) + 1;
    if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
    if (Lk>Li) { cerr << progstr+": " << __LINE__ << errstr << "Li (length of input vecs) must be >= Lk" << endl; return 1; }
    if (pad<=-(int)Li) { cerr << progstr+": " << __LINE__ << errstr << "pad length must be > -Li" << endl; return 1; }
    if (pm && pad>(int)Li) { cerr << progstr+": " << __LINE__ << errstr << "Li (length of input vecs) must be >= pad length" << endl; return 1; }
    if (Ti<(int)Lk) { cerr << progstr+": " << __LINE__ << errstr << "Li+2*pad must be >= Lk" << endl; return 1; }
    if (Ti<Tk) { cerr << progstr+": " << __LINE__ << errstr << "Li+2*pad must be >= dil*(Lk-1)" << endl; return 1; }


    //Set output header info
    Lo = 1u + size_t(Ti-Tk)/str + size_t(ceil_mode && size_t(Ti-Tk)%str);
    o1.F = i1.F; o1.T = i1.T;
    o1.R = i1.iscolmajor() ? N : Lo;
    o1.C = i1.iscolmajor() ? Lo : N;
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


    //Process
    if (i1.T==1u)
    {
        float *X, *Y;
        try { X = new float[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
        try { Y = new float[o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
        if (codee::avgpool1d_s(Y,X,N,Li,Lk,pad,str,dil,ceil_mode,pm))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] Y;
    }
    else if (i1.T==2u)
    {
        double *X, *Y;
        try { X = new double[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
        try { Y = new double[o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
        if (codee::avgpool1d_d(Y,X,N,Li,Lk,pad,str,dil,ceil_mode,pm))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] Y;
    }
    else if (i1.T==101u)
    {
        float *X, *Y;
        try { X = new float[2u*i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
        try { Y = new float[2u*o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
        if (codee::avgpool1d_c(Y,X,N,Li,Lk,pad,str,dil,ceil_mode,pm))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] Y;
    }
    else if (i1.T==102u)
    {
        double *X, *Y;
        try { X = new double[2u*i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
        try { Y = new double[2u*o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
        if (codee::avgpool1d_z(Y,X,N,Li,Lk,pad,str,dil,ceil_mode,pm))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] Y;
    }
    else
    {
        cerr << progstr+": " << __LINE__ << errstr << "data type not supported" << endl; return 1;
    }
    

    //Exit
    return ret;
}
