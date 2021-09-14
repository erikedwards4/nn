//@author Erik Edwards
//@date 2018-present
//@license BSD 3-clause


#include <ctime>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string>
#include <cstring>
#include <valarray>
#include <unordered_map>
#include <argtable2.h>
#include "../util/cmli.hpp"
#include "linear.c"

#ifdef I
#undef I
#endif


int main(int argc, char *argv[])
{
    using namespace std;
    timespec tic, toc;


    //Declarations
    int ret = 0;
    const string errstr = ": \033[1;31merror:\033[0m ";
    const string warstr = ": \033[1;35mwarning:\033[0m ";
    const string progstr(__FILE__,string(__FILE__).find_last_of("/")+1,strlen(__FILE__)-string(__FILE__).find_last_of("/")-5);
    const valarray<size_t> oktypes = {1u,2u,101u,102u};
    const size_t I = 2u, O = 1u;
    ifstream ifs1, ifs2; ofstream ofs1;
    int8_t stdi1, stdi2, stdo1, wo1;
    ioinfo i1, i2, o1;
    size_t Ni, No, L;


    //Description
    string descr;
    descr += "IN method.\n";
    descr += "Linear transformation (weights only) of Ni inputs to No outputs.\n";
    descr += "\n";
    descr += "Input X has Ni neurons and output Y has No neurons.\n";
    descr += "\n";
    descr += "The Ni or No neurons are always contiguous in memory, such that:\n";
    descr += "\n";
    descr += "If col-major: Y[:,l] = W' * X[:,l] \n";
    descr += "where:\n";
    descr += "X has size Ni x L \n";
    descr += "Y has size No x L \n";
    descr += "W has size Ni x No \n";
    descr += "\n";
    descr += "If row-major: Y[l,L] = X[l,:] * W' \n";
    descr += "where:\n";
    descr += "X has size L x Ni \n";
    descr += "Y has size L x No \n";
    descr += "W has size No x Ni \n";
    descr += "\n";
    descr += "Note that W is transposed (non-conjugate), \n";
    descr += "such that vecs of length Ni are contiguous in memory.\n";
    descr += "\n";
    descr += "Examples:\n";
    descr += "$ linear X W -o Y \n";
    descr += "$ linear X W > Y \n";
    descr += "$ cat X | linear - W > Y \n";


    //Argtable
    int nerrs;
    struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,W)");
    struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");
    struct arg_lit *a_help = arg_litn("h","help",0,1,"display this help and exit");
    struct arg_end  *a_end = arg_end(5);
    void *argtable[] = {a_fi, a_fo, a_help, a_end};
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
    stdi2 = (a_fi->count<=1 || strlen(a_fi->filename[1])==0u || strcmp(a_fi->filename[1],"-")==0);
    if (stdi1+stdi2>1) { cerr << progstr+": " << __LINE__ << errstr << "can only use stdin for one input" << endl; return 1; }
    if (stdi1+stdi2>0 && isatty(fileno(stdin))) { cerr << progstr+": " << __LINE__ << errstr << "no stdin detected" << endl; return 1; }


    //Check stdout
    if (a_fo->count>0) { stdo1 = (strlen(a_fo->filename[0])==0u || strcmp(a_fo->filename[0],"-")==0); }
    else { stdo1 = (!isatty(fileno(stdout))); }
    wo1 = (stdo1 || a_fo->count>0);


    //Open inputs
    if (stdi1) { ifs1.copyfmt(cin); ifs1.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs1.open(a_fi->filename[0]); }
    if (!ifs1) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 1" << endl; return 1; }
    if (stdi2) { ifs2.copyfmt(cin); ifs2.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs2.open(a_fi->filename[1]); }
    if (!ifs2) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 2" << endl; return 1; }


    //Read input headers
    if (!read_input_header(ifs1,i1)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 1" << endl; return 1; }
    if (!read_input_header(ifs2,i2)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 2" << endl; return 1; }
    if ((i1.T==oktypes).sum()==0 || (i2.T==oktypes).sum()==0)
    {
        cerr << progstr+": " << __LINE__ << errstr << "input data type must be in " << "{";
        for (auto o : oktypes) { cerr << int(o) << ((o==oktypes[oktypes.size()-1u]) ? "}" : ","); }
        cerr << endl; return 1;
    }


    //Get options


    //Checks
    if (i1.T!=i2.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have the same data type" << endl; return 1; }
    if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
    if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (W) found to be empty" << endl; return 1; }
    if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) must be a matrix" << endl; return 1; }
    if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (W) must be a matrix" << endl; return 1; }
    L = i1.iscolmajor() ? i1.C : i1.R;
    Ni = i2.iscolmajor() ? i2.R : i2.C;
    No = i2.iscolmajor() ? i2.C : i2.R;
    if (i1.iscolmajor())
    {
        if (i1.R!=Ni) { cerr << progstr+": " << __LINE__ << errstr << "Input 1 (X) must have size Ni x L for col-major" << endl; return 1; }
    }
    else
    {
        if (i1.C!=Ni) { cerr << progstr+": " << __LINE__ << errstr << "Input 1 (X) must have size L x Ni for row-major" << endl; return 1; }
    }


    //Set output header info
    o1.F = i1.F; o1.T = i1.T;
    o1.R = i1.iscolmajor() ? No : i1.R;
    o1.C = i1.isrowmajor() ? No : i1.C;
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
    clock_gettime(CLOCK_REALTIME,&tic);
    if (i1.T==1u)
    {
        float *X, *W, *Y;
        try { X = new float[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
        try { W = new float[i2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (W)" << endl; return 1; }
        try { Y = new float[o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
        try { ifs2.read(reinterpret_cast<char*>(W),i2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (W)" << endl; return 1; }
        if (codee::linear_s(Y,X,W,Ni,No,L))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] W; delete[] Y;
    }
    else if (i1.T==2u)
    {
        double *X, *W, *Y;
        try { X = new double[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
        try { W = new double[i2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (W)" << endl; return 1; }
        try { Y = new double[o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
        try { ifs2.read(reinterpret_cast<char*>(W),i2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (W)" << endl; return 1; }
        if (codee::linear_d(Y,X,W,Ni,No,L))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] W; delete[] Y;
    }
    else if (i1.T==101u)
    {
        float *X, *W, *Y;
        try { X = new float[2u*i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
        try { W = new float[2u*i2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (W)" << endl; return 1; }
        try { Y = new float[2u*o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
        try { ifs2.read(reinterpret_cast<char*>(W),i2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (W)" << endl; return 1; }
        if (codee::linear_c(Y,X,W,Ni,No,L))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] W; delete[] Y;
    }
    else if (i1.T==102u)
    {
        double *X, *W, *Y;
        try { X = new double[2u*i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
        try { W = new double[2u*i2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (W)" << endl; return 1; }
        try { Y = new double[2u*o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
        try { ifs2.read(reinterpret_cast<char*>(W),i2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (W)" << endl; return 1; }
        if (codee::linear_z(Y,X,W,Ni,No,L))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] W; delete[] Y;
    }
    else
    {
        cerr << progstr+": " << __LINE__ << errstr << "data type not supported" << endl; return 1;
    }
    clock_gettime(CLOCK_REALTIME,&toc);
    cerr << "elapsed time = " << double(toc.tv_sec-tic.tv_sec)*1e3 + double(toc.tv_nsec-tic.tv_nsec)/1e6 << " ms" << endl;
    

    //Exit
    return ret;
}
