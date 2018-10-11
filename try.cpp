#include <arrayfire.h>
#include <iostream>
// #include <af/array.h>
#include <vector>
using namespace af;
int main()
{
    array a = randu(3,300), b = randu(3,200);
    std::cout << a.dims() << " & " << b.dims() << "\n";
    int alen = a.dims(1), blen = b.dims(1), feat_len = a.dims(0);
    
    b = moddims(b, feat_len, 1, blen);
    std::cout << a.dims() << " & " << b.dims() << "\n";

    a = tile(a, 1, 1, blen);
    b = tile(b, 1, alen, 1);
    std::cout << a.dims() << " & " << b.dims() << "\n";

    array dist = abs(a - b);
    std::cout << dist.dims() << "\n";

    array dist0 = sum(dist, 0), dist1 = sum(dist, 1);
    std::cout << dist0.dims() << " & " << dist1.dims() << "\n";

}