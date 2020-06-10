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
#include "split2.c"

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
    const valarray<uint8_t> oktypes = {1,2,101,102};
    const size_t I = 1, O = 2;
    ifstream ifs1; ofstream ofs1, ofs2;
    int8_t stdi1, stdo1, stdo2, wo1, wo2;
    ioinfo i1, o1, o2;
    int dim;


    //Description
    string descr;
    descr += "Splits 1 input X into 2 equal-sized outputs Y1, Y2.\n";
    descr += "\n";
    descr += "Use -d (--dim) to give the dimension (axis) [default=0].\n";
    descr += "Use -d0 to work along cols --> Y has size R/2 x C.\n";
    descr += "Use -d1 to work along rows --> Y has size R x C/2.\n";
    descr += "\n";
    descr += "For dim=0, num rows X must be even (R%2==0).\n";
    descr += "For dim=1, num cols X must be even (C%2==0).\n";
    descr += "\n";
    descr += "Examples:\n";
    descr += "$ split2 X -o Y1 -o Y2 \n";
    descr += "$ split2 X -o Y1 > Y2 \n";
    descr += "$ split2 -d1 X -o Y1 -o Y2 \n";
    descr += "$ cat X | split2 -o Y1 -o Y2 \n";


    //Argtable
    int nerrs;
    struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
    struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension [default=0]");
    struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output files (Y1,Y2)");
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
    if (a_fo->count>1) { stdo2 = (strlen(a_fo->filename[1])==0 || strcmp(a_fo->filename[1],"-")==0); }
    else { stdo2 = (!isatty(fileno(stdout)) && a_fo->count==1 && stdo1==0); }
    if (stdo1+stdo2>1) { cerr << progstr+": " << __LINE__ << errstr << "can only use stdout for one output" << endl; return 1; }
    wo1 = (stdo1 || a_fo->count>0); wo2 = (stdo2 || a_fo->count>1);


    //Open input
    if (stdi1) { ifs1.copyfmt(cin); ifs1.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs1.open(a_fi->filename[0]); }
    if (!ifs1) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file" << endl; return 1; }


    //Read input header
    if (!read_input_header(ifs1,i1)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file" << endl; return 1; }
    if ((i1.T==oktypes).sum()==0)
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
    if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) found to be empty" << endl; return 1; }
    if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) must be a matrix" << endl; return 1; }
    if (dim==0 && i1.R%2) { cerr << progstr+": " << __LINE__ << errstr << "num rows X must be even for dim=0" << endl; return 1; }
    if (dim==1 && i1.C%2) { cerr << progstr+": " << __LINE__ << errstr << "num cols X must be even for dim=1" << endl; return 1; }


    //Set output header infos
    o1.F = o2.F = i1.F;
    o1.T = o2.T = i1.T;
    o1.R = o2.R = (dim==0) ? i1.R/2 : i1.R;
    o1.C = o2.C = (dim==1) ? i1.C/2 : i1.C;
    o1.S = o2.S = i1.S;
    o1.H = o2.S = i1.H;


    //Open outputs
    if (wo1)
    {
        if (stdo1) { ofs1.copyfmt(cout); ofs1.basic_ios<char>::rdbuf(cout.rdbuf()); } else { ofs1.open(a_fo->filename[0]); }
        if (!ofs1) { cerr << progstr+": " << __LINE__ << errstr << "problem opening output file 1" << endl; return 1; }
    }
    if (wo2)
    {
        if (stdo2) { ofs2.copyfmt(cout); ofs2.basic_ios<char>::rdbuf(cout.rdbuf()); } else { ofs2.open(a_fo->filename[1]); }
        if (!ofs2) { cerr << progstr+": " << __LINE__ << errstr << "problem opening output file 2" << endl; return 1; }
    }


    //Write output headers
    if (wo1 && !write_output_header(ofs1,o1)) { cerr << progstr+": " << __LINE__ << errstr << "problem writing header for output file 1" << endl; return 1; }
    if (wo2 && !write_output_header(ofs2,o2)) { cerr << progstr+": " << __LINE__ << errstr << "problem writing header for output file 2" << endl; return 1; }


    //Other prep


    //Process
    if (i1.T==1)
    {
        float *X, *Y1, *Y2;
        try { X = new float[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
        try { Y1 = new float[o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 1 (Y1)" << endl; return 1; }
        try { Y2 = new float[o2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 2 (Y2)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
        if (openn::split2_s(Y1,Y2,X,int(i1.R),int(i1.C),dim,i1.iscolmajor()))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y1),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 1 (Y1)" << endl; return 1; }
            try { ofs2.write(reinterpret_cast<char*>(Y2),o2.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 2 (Y2)" << endl; return 1; }
        }
        delete[] X; delete[] Y1; delete[] Y2;
    }
    else if (i1.T==2)
    {
        double *X, *Y1, *Y2;
        try { X = new double[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
        try { Y1 = new double[o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 1 (Y1)" << endl; return 1; }
        try { Y2 = new double[o2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 2 (Y2)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
        if (openn::split2_d(Y1,Y2,X,int(i1.R),int(i1.C),dim,i1.iscolmajor()))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y1),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 1 (Y1)" << endl; return 1; }
            try { ofs2.write(reinterpret_cast<char*>(Y2),o2.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 2 (Y2)" << endl; return 1; }
        }
        delete[] X; delete[] Y1; delete[] Y2;
    }
    else if (i1.T==101)
    {
        float *X, *Y1, *Y2;
        try { X = new float[2u*i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
        try { Y1 = new float[2u*o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 1 (Y1)" << endl; return 1; }
        try { Y2 = new float[2u*o2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 2 (Y2)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
        if (openn::split2_c(Y1,Y2,X,int(i1.R),int(i1.C),dim,i1.iscolmajor()))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y1),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 1 (Y1)" << endl; return 1; }
            try { ofs2.write(reinterpret_cast<char*>(Y2),o2.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 2 (Y2)" << endl; return 1; }
        }
        delete[] X; delete[] Y1; delete[] Y2;
    }
    else if (i1.T==102)
    {
        double *X, *Y1, *Y2;
        try { X = new double[2u*i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
        try { Y1 = new double[2u*o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 1 (Y1)" << endl; return 1; }
        try { Y2 = new double[2u*o2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 2 (Y2)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
        if (openn::split2_z(Y1,Y2,X,int(i1.R),int(i1.C),dim,i1.iscolmajor()))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y1),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 1 (Y1)" << endl; return 1; }
            try { ofs2.write(reinterpret_cast<char*>(Y2),o2.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 2 (Y2)" << endl; return 1; }
        }
        delete[] X; delete[] Y1; delete[] Y2;
    }
    else
    {
        cerr << progstr+": " << __LINE__ << errstr << "data type not supported" << endl; return 1;
    }
    

    //Exit
    return ret;
}

