#include <stdio.h>
#include <string>
#include <iostream>
using namespace std;
void cha(string& s, int a, int b){
    while(a<b){
        char tmp=s[b];
        s[b] = s[a];
        s[a] = tmp;
        a++;b--;
    }
}
int main(){
    string s;
    cin>>s;
    int len;
    cin>>len;
    cha(s,0,len-1);
    cha(s,len,s.size()-1);
    cha(s,0,s.size()-1);
    cout<<s<<endl;
    return 0;
}


