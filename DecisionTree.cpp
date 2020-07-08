/*
 * @Description: C++ʵ�־�����
 * @Author: szq
 * @Github: https://github.com/MrQqqq
 * @Date: 2020-06-29 01:31:54
 * @LastEditors: szq
 * @LastEditTime: 2020-07-08 18:57:10
 * @FilePath: \cpp\src\DecisionTree\DecisionTree.cpp
 */ 

#include<iostream>
#include<math.h>
#include<algorithm>
#include"DecisionTree.h"
using namespace std;

/**
 * @destription: ��������
 * @param {type} trainData��ѵ������  trainLabel��ѵ����ǩ
 * @return: û�з���ֵ
 */
void DecisionTree::loadData(vector<vector<int>> &trainData,vector<int> &trainLabel){
    //��������������������������ݼ���ǩ��������һ����ʱ������������
    if(trainData.size() != trainLabel.size()){
        cerr << "input error" << endl;
        return;
    }
    //��ʼ��
    this->trainData = trainData;
    this->trainLabel = trainLabel;

    //����featureValues
    for(auto data : trainData){
        for(int i = 0;i < data.size();++i){
            featureValues[i].insert(data[i]);
        }
    }
}

/**
 * @destription: ͳ�����ݼ���ÿ����ǩ��������������Ϊ1�������ͽ��Ϊ2������
 * @param {type} dataset�����ݼ�����ѵ�����ݵ��Ӽ������������ʾ��ÿһ��������ʾ�ڼ���ѵ������
 * @return: ��ǩ������ǩ����map
 */
map<int,int> DecisionTree::labelCount(vector<int> &dataset){
    map<int,int> res;
    //�������ݼ���ͳ�Ʊ�ǩ���ֵĴ���
    for(int index : dataset){
        res[trainLabel[index]]++;
    }
    return res;
}

/**
 * @destription: ������Ϣ�أ���ʽΪ-pi * log(pi)���
 * @param {type} ��dataset�����ݼ�
 * @return: ��Ϣ��
 */

double DecisionTree::caculateEntropy(vector<int> &dataset){
    map<int,int> label_count = labelCount(dataset);
    int len = dataset.size();
    double result = 0;
    for(auto count : label_count){
        double pi = count.second / static_cast<double>(len);
        result -= pi * log2(pi);
    }
    return result;
}

/**
 * @destription: �����������������Ӽ�
 * @param {type} dataset�����ݼ�
 * @param feature:�ڼ�������
 * @param value:�ڼ�������������ֵ
 * @return: ���ػ��ֵ��Ӽ�
 */    
vector<int> DecisionTree::splitDataset(vector<int> &dataset,int &feature,int &value){
    vector<int> res;
    for(int index : dataset){
        if(trainData[index][feature] == value){
            res.push_back(index);
        }
    }
    return res;
}

/**
 * @destription: ������Ϣ��
 * @param {type} 
 * @return: 
 */
double DecisionTree::caculateGain(vector<int> &dataset,int &feature){
    set<int> values = featureValues[feature];
    double result = 0;
    for(int value : values){
        vector<int> subDataset = splitDataset(dataset,feature,value);
        result += subDataset.size() / static_cast<double>(dataset.size()) * caculateEntropy(subDataset);
    }
    return caculateEntropy(dataset) - result;

}

/**
 * @destription: ��ȡ��ǩͳ���г��ִ������ı�ǩ
 * @param labelCount:��ǩͳ��
 * @return: ���س��ִ������ı�ǩ��
 */
int DecisionTree::getMaxTimesLabel(map<int,int> &labelCount){
    int max_count = 0;
    int res;
    for(auto label : labelCount){
        if(max_count <= label.second){
            max_count = label.second;
            res = label.first;
        }
    }
    return res;
}

/**
 * @destription: ��ȡ��������Ϣ������������Ϣ���������Ӧ������
 * @param gains:����������Ϣ����
 * @return: �����Ϣ�����Ӧ������
 */
int DecisionTree::getMaxGainFeature(map<int,double> &gains){
    double max_gain = 0;
    int max_gain_feature;
    for(auto gain : gains){
        if(max_gain <= gain.second){
            max_gain = gain.second;
            max_gain_feature = gain.first;
        }
    }
    return max_gain_feature;
}

/**
 * @destription: ����������
 * @param dataset:���ݼ�
 * @param features:������ 
 * @return: ���ؾ��������ڵ�ָ��
 */
TreeNode* DecisionTree::createTree(vector<int> &dataset,vector<int> &features){
    TreeNode *root = new TreeNode();
    map<int,int> label_count = labelCount(dataset);
    //���������Ϊ�գ������Ϊ���ڵ��������Ϊ��ǩ�г��ִ������ı�ǩ
    if(features.size() == 0){
        root->result = getMaxTimesLabel(label_count);
        root->isLeaf = true;
        return root;
    }
    //������ݼ���ֻ����һ�ֱ�ǩ�������Ϊ���ڵ��������Ϊ�ñ�ǩ
    if(label_count.size() == 1){
        root->result = label_count.begin()->first;
        root->isLeaf = true;
        return root;
    }

    //������������ÿ����������Ϣ����
    map<int,double> gains;
    for(int feature : features){
        gains[feature] = caculateGain(dataset,feature);
    }

    //��ȡ�����Ϣ�����������������Ϣ����
    int max_gain_feature = getMaxGainFeature(gains);
    vector<int> subFeatures = features;
    subFeatures.erase(find(subFeatures.begin(),subFeatures.end(),max_gain_feature));
    for(int value : featureValues[max_gain_feature]){
        TreeNode *branch = new TreeNode();//������֧
        vector<int> subDataset = splitDataset(dataset,max_gain_feature,value);
        
        
        //����Ӽ�Ϊ�գ�����֧�ڵ���ΪҶ�ڵ㣬���Ϊ��ǩ�г��ִ������ı�ǩ
        if(subDataset.size() == 0){
            branch->isLeaf = true;
            branch->result = getMaxTimesLabel(label_count);
            branch->attr = max_gain_feature;
            branch->attr_value = value;
            root->branchs.push_back(branch);
        }
        //����ݹ鴴����
        else{
            branch = createTree(subDataset,subFeatures);
            branch->attr = max_gain_feature;
            branch->attr_value = value;
            root->branchs.push_back(branch);
        }
    }
    return root;
}

/**
 * @destription: ���캯��
 * @param trainData:ѵ������
 * @param trainLabel:ѵ�����ݱ�ǩ
 * @param threshold:��ֵ
 * @return: û�з���ֵ
 */
DecisionTree::DecisionTree(vector<vector<int>> &trainData,vector<int> &trainLabel,int &threshold){
    loadData(trainData,trainLabel);//��������
    this->threshold = threshold;//������ֵ
    vector<int> dataset(trainData.size());//���ݼ�
    for(int i = 0;i < trainData.size();i++){
        dataset[i] = i;
    }
    vector<int> features(trainData[0].size());//���Լ���
    for(int i = 0;i < trainData[0].size();i++){
        features[i] = i;
    }
    decisionTreeRoot = createTree(dataset,features);//����������
}

/**
 * @destription: ����
 * @param testData:�������� 
 * @param root:���������ڵ�
 * @return: ���ط�����
 */
int DecisionTree::classify(vector<int> &testData,TreeNode *root){
    //����������ڵ���Ҷ�ӽڵ㣬ֱ�ӷ��ؽ��
    if(root->isLeaf){
        return root->result;
    }
    for(auto node : root->branchs){
        //�ҵ���֧�����ڷ�֧����ϸ��
        if(testData[node->attr] == node->attr_value){
            return classify(testData,node);
        }
    }
    return 0;
}


int main(){
    //ѵ������
    vector<vector<int>> trainData = {
        {0, 0, 0, 0},
        {0, 0, 0, 1},
        {0, 1, 0, 1},
        {0, 1, 1, 0},
        {0, 0, 0, 0},
        {1, 0, 0, 0},
        {1, 0, 0, 1},
        {1, 1, 1, 1},
        {1, 0, 1, 2},
        {1, 0, 1, 2},
        {2, 0, 1, 2},
        {2, 0, 1, 1},
        {2, 1, 0, 1},
        {2, 1, 0, 2},
        {2, 0, 0, 0}
    };
    //ѵ����ǩ
    vector<int> trainLabel = {0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0};
    int threshold = 0;
    DecisionTree dt = DecisionTree(trainData,trainLabel,threshold);
    
    //��������
    vector<int> testData = {2,1,1,1};
    TreeNode *root = dt.decisionTreeRoot;
    int type = dt.classify(testData,root);
    cout << type << endl;
    return 0;
}