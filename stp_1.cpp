#include<bits/stdc++.h>
using namespace std;
#define INF INT_MAX

void computeShortestTimes(int vCount, int eCount, int *from, int *to, int *departure, int *duration, int *&shortesttime,int source, int esdg_edge,int *esdg_from,int *esdg_to)
{	
    struct node
    {
        int first;
        int second;

        int third;
        int fourth;
    };
	cout<<"hi"<<endl;
	/*for(int i=0; i<eCount; i++){
       // int temp;
		cout<<" "<<from[i]<<" "<<to[i]<<" "<<departure[i]<<" "<<duration[i]<<endl;
	}
	cout<<"#####"<<endl;
*/

    vector<node>temp_node(eCount);//number of edges

	for(int i=0;i<eCount;i++)
	{
	    temp_node[i].first = from[i];
	    temp_node[i].second =  to[i];
	    temp_node[i].third= departure[i];
	    temp_node[i].fourth= duration[i];
	}
	/*cout<<"line 24"<<endl;
	for(int i=0;i<eCount;i++)
	{ 
		
	    cout<<temp_node[i].first<<endl;
	}*/

	stack<int>s;
	vector<vector<int>>graph(esdg_edge);
	//cout<<" "<<esdg_edge<<" "<<vCount<<" "<<eCount<<endl;
	vector<int>shortest_path(vCount,INT_MAX);
	shortest_path[0]=0;
	for(int i=0;i<esdg_edge;i++)
	{
		int x,y;
		x=esdg_from[i];
		y=esdg_to[i];
		//<<x<<"-->"<<y<<endl;

		graph[x].push_back(y);
		for(auto temp:temp_node){
			if( x == 0 && temp.first == x )
			{
				//cout<<"--"<<temp.second<<endl;
				s.push(x);
			}
		}
	}

	/*for(auto x:graph)
	{   

		for(auto y:x)
		{
			cout<<y<<endl;
		}
	} */

	while(s.empty()==false)
	{
		int e = s.top();
        s.pop();
		for(auto temp:temp_node){

			if(temp.first == e){

				shortest_path[temp_node[e].second]=
				min(shortest_path[temp_node[e].second],(temp_node[e].fourth)
				+shortest_path[temp_node[e].first]);
        
			}
		}
		for(auto x:graph[e])
		{
			s.push(x);
		} 
	}
	for(auto x: shortest_path){
		cout<<x<<endl;
	}
}

void readInput_1(char *fileName_1, int &vCount, int &eCount, int *&from, int *&to, int *&departure, int *&duration)
{
	ifstream fin;
	fin.open(fileName_1);
	fin>>vCount>>eCount;

	//tuple of edge
	from = new int[eCount];
	to = new int[eCount];
	departure = new int[eCount];
	duration = new int[eCount];

	for(int i=0; i<=eCount-1; i++){
       // int temp;
		fin>>from[i]>>to[i]>>departure[i]>>duration[i]; 
	}

	for(int i=0; i<eCount; i++){
       // int temp;
		cout<<" "<<from[i]<<" "<<to[i]<<" "<<departure[i]<<" "<<duration[i]<<endl;
	}
	cout<<"##### line 95"<<endl;
}

void readInput_2(char *fileName_2, int &esdg_vertice, int &esdg_edge, int *&esdg_from, int *&esdg_to, int *&esdg_link, int *&esdg_value, int &count)
{
	ifstream fin;
	fin.open(fileName_2);
	fin>>esdg_vertice>>esdg_edge;

	esdg_link = new int[esdg_vertice];
	esdg_value = new int[esdg_edge];
	
	for(int i=0; i<esdg_vertice; i++){
		fin>>esdg_link[i];
	}

	for(int i=0;i<esdg_edge;i++){
		fin>>esdg_value[i];
	}
	//esdg_vertice = esdg_vertice-1;
	esdg_from = new int[esdg_edge];
	esdg_to = new int[esdg_edge];

	count=0;
	for(int vertices_number=0;vertices_number<esdg_vertice-1;vertices_number++)
	{
		int from = esdg_link[vertices_number];
		int to = esdg_link[vertices_number+1];

		for(int j=from;j<to;j++){
			esdg_from[count] = vertices_number;
			esdg_to[count] = esdg_value[j];
			count++;
		}
	}
	for(int i=0;i<count;i++){
		cout<<esdg_from[i]<<" "<<esdg_to[i]<<endl;
	}
	esdg_edge = count;
	
}

void createHostMemory(int *&shortesttime,int vCount)
{
	shortesttime = new int[vCount];
}

int main(int argc, char *argv[])
{
	int vCount, eCount, qCount;
	int *from, *to, *departure, *duration, *shortesttime;
	int *source, *departureTime,*starting_time, *ending_time;
	char fileName_1[100];
    char fileName_2[100];
  
	
	strcpy(fileName_1, argv[1]);//edges stream
    strcpy(fileName_2, argv[2]);//esdg data
    
    int esdg_vertice,esdg_edge,count;
    int *esdg_from,*esdg_to, *esdg_link, *esdg_value;

	readInput_1(fileName_1,vCount, eCount, from, to, departure, duration);
    readInput_2(fileName_2,esdg_vertice, esdg_edge,esdg_from,esdg_to,esdg_link,esdg_value,count);
	esdg_vertice = eCount;
	
	cout<<eCount<<" "<<count<<endl;

	for(int i=0; i<1; i++)
	{
		computeShortestTimes(vCount,eCount,from,to,departure,duration,shortesttime,0,esdg_edge,esdg_from,esdg_to);
	}
	return 0;
}