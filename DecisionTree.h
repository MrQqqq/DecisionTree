/*
 * @Description: 决策树头文件
 * @Author: szq
 * @Github: https://github.com/MrQqqq
 * @Date: 2020-07-08 16:36:21
 * @LastEditors: szq
 * @LastEditTime: 2020-07-08 18:51:47
 * @FilePath: \cpp\src\DecisionTree\DecisionTree.h
 */ 
#include<vector>
#include<map>
#include<set>
using namespace std;

//决策树节点结构
struct TreeNode{
    bool isLeaf = false;//是否是叶子节点
    int result = -1;//如果是叶子节点的话，对应的label索引
    vector<TreeNode*> branchs;//分支节点
    int attr = -1;//特征
    int attr_value = -1;//特征值
};

class DecisionTree{
public:

    vector<vector<int>> trainData;//训练数据集的特征数据
    vector<int> trainLabel;//训练数据集对应的标签
    map<int,set<int>> featureValues;//每个特征的类别
    float threshold;//阈值
    TreeNode *decisionTreeRoot;//决策树的根节点
    
    DecisionTree(vector<vector<int>> &trainData,vector<int> &trainLabel,int &threshold);
    void loadData(vector<vector<int>> &trainData,vector<int> &trainLabel);// 导入数据
    map<int,int> labelCount(vector<int> &dataset);//统计数据集中每个标签的数量，比如结果为1的数量和结果为2的数量
    double caculateEntropy(vector<int> &dataset);//计算信息熵
    vector<int> splitDataset(vector<int> &dataset,int &feature,int &value);//分割数据集
    double caculateGain(vector<int> &dataset,int &feature);//计算信息增益
    int getMaxTimesLabel(map<int,int> &labelCount);//获取出现次数最多的标签
    int getMaxGainFeature(map<int,double> &gains);//获取最大信息增益的特征
    TreeNode* createTree(vector<int> &dataset,vector<int> &features);//创建决策树
    int classify(vector<int> &testData,TreeNode *root);
};