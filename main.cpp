//
//  main.cpp
//  NoLinearClassification
//
//  Created by 刘建安 on 5/1/15.
//  Copyright (c) 2015 刘建安. All rights reserved.
//

#include <iostream>
#include "memory.h"
#include "map"
#include "vector"
#include "fstream"
#include "math.h"
#include "string"
#include "stdio.h"
#include "stdlib.h"
using namespace std;

//总共有3层，其中包括一层输入层，一层隐藏层和一层输出层
//第一层有618个节点，第二层有100个节点，第三层有26个节点
map<int, pair<vector<double>, vector<double> > >samples; //训练样本 map<id, pair<values of features, references>>
#define numOfFeatures 618 //属性个数
size_t numOfSamples = 1; //样本数量/输入层节点数
#define level2  100 //第二层节点数目
#define level3  26 //第三层/输出层节点数目
//保存隐藏层节点的值
double a2[level2];
double a3[level3];
//保存隐藏层节点的值
double z2[level2];
double z3[level3];

//保存误差值
double Error1[level2][numOfFeatures];
double Error2[level3][level2];
//保存各层的权重
double theta1[level2][numOfFeatures];
double theta2[level3][level2];
//其他参数
double a = 1e-2; //learning rate
int trainingSize = 100; //训练循环次数

//用于读取训练样本
void readTrainSamples(string fileName){
    ifstream in("train.csv");
    if (!in.is_open()){
        cout << "open file " << fileName << " failed" <<endl;
        exit(-1);
    }
    cout << "reading samples..." << endl;
    
    string feature;
    getline(in, feature); //去掉第一行的标签
    
    
    int id = 4;
    double value = 0; //保存属性的值
    int label = 0;  //保存真实的值（y值）
    char dot = ' '; //保存逗号
    
    //读取样本
    while(1){
        in >> id;
        if(in.eof())
            break;
        
        samples[id].first.clear(); //防止有相同的ID，而重复采样
        //读取所有feature的值
        samples[id].first.push_back(1);
        for (int i = 1; i < numOfFeatures; i++){
            in >> dot; //忽略逗号
            in >> value; //一个属性的值
            samples[id].first.push_back(value);
        }
        in >> dot;
        in >> label; //读取reference
        
        for (int i = 1; i <= 26; i++){
            if (i == label)
                samples[id].second.push_back(1);
            else
                samples[id].second.push_back(0);
        }
        while(in.peek() == '\n')
            in.get();
    }
    in.close();
    numOfSamples = samples.size();
    
    cout << "read samples completed" << endl;
}






void initialize(){
    for (int i = 0; i < level2 ; i++) {
        for (int j = 0; j < numOfFeatures; j ++){
            //theta1[i][j] = ((rand() * 1.0 / RAND_MAX) - 0.5) / 10.0;
            theta1[i][j] = rand() * 1.0 / RAND_MAX * sqrt(2.0 / 385);
        }
    }
    
    for (int i = 0; i < level3; i++){
        for (int j = 0; j < level2; j++){
            //theta2[i][j] = ((rand() * 1.0 / RAND_MAX) - 0.5) / 10.0;
            
            theta2[i][j] = rand() * 1.0 / RAND_MAX * sqrt(2.0 / 100);
        }
    }
    
}


void training(){
    
    cout << "training ..." << endl;
    int tmpSize = trainingSize;
    
    while (tmpSize--) {
        
        cout << tmpSize << endl;
        map<int, pair<vector<double>, vector<double> > >::iterator it = samples.begin();
        
        for (; it != samples.end(); it++){
            double tmp = 0;
            //计算第二层各个节点的值
            a2[0] = 1;
            for (int i = 1; i < level2; i++){
                tmp = 0;
                for (int j = 0; j < numOfFeatures; j++){
                    tmp += (theta1[i][j] * it->second.first[j]);
                }
                z2[i] = tmp;
                a2[i] = 1.0 / (1.0 + exp(-tmp));
            }
            
            //计算第三层各节点的值
            
            for (int i = 0; i < level3; i++){
                tmp = 0;
                for (int j = 0; j < level2; j++){
                    tmp += (theta2[i][j] * a2[j]);
                }
                z3[i] = tmp;
                a3[i] = 1.0 / (1.0 + exp(-tmp));
            }
            
            
            for (int i = 0; i < level2; i++){
                for (int j = 0; j < level3; j++){
                    Error2[j][i] = (a3[j] - it->second.second[j]) * a3[j] * (1 - a3[j]) * a2[i];
                }
            }
            
            for (int i = 0; i < numOfFeatures; i++){
                for (int j = 1; j < level2; j++){
                    tmp = 0;
                    for (int k = 0; k < level3; k++)
                        tmp += (a3[k] - it->second.second[k]) * a3[k] * (1 - a3[k]) * theta2[k][j] * a2[j] * (1 - a2[j]) * it->second.first[i];
                    //cout << Error1[j][i] << endl;
                    Error1[j][i] = tmp;
                }
                
            }
            
            
            for (int i = 0; i < level3; i++){
                for (int j = 0; j < level2; j++){
                    theta2[i][j] = theta2[i][j] - a * Error2[i][j];
                }
            }
            
            for (int i = 1; i < level2; i++){
                for (int j = 0; j < numOfFeatures; j++){
                    theta1[i][j] = theta1[i][j] - a * Error1[i][j];
                }
            }
            
            
        }
    }
    
    cout << "trainig completed" << endl;
}

void predict(){
    
    
    ifstream in("test.csv"); //读取测试文件
    ofstream out("result.csv"); //结果文件
    if (!in.is_open()){
        cout << "open file test" << " failed" <<endl;
        exit(-1);
    }
    if (!out.is_open()){
        cout << "open result file failed" << endl;
        exit(-1);
    }
    
    cout << "predicting ..." << endl;
    out << "id,label" << "\n";
    
    string feature;
    getline(in, feature); //去掉第一行的标签
    
    int id = 0;
    char dot = ' ';
    while(1){
        double features[numOfFeatures];
        in >> id;
        if(in.eof())
            break;
        //读取384个属性值
        features[0] = 1;
        for (int i = 1; i < numOfFeatures; i++){
            in >> dot;
            in >> features[i];
            //cout << features[i] << endl;
        }
        //计算第二层各节点的值
        a2[0] = 1;
        for (int i = 1;  i < level2; i ++){
            double tmp = 0;
            for (int j = 0; j < numOfFeatures; j++){
                tmp += (theta1[i][j] * features[j]);
            }
            a2[i] = 1.0 / (1.0 + exp(-tmp));
        }
        //计算第三层各节点的值
        for (int i = 0; i < level3; i++){
            double tmp = 0;
            for (int j = 0; j < level2; j++){
                tmp += (theta2[i][j] * a2[j]);
            }
            //cout << tmp << endl;
            a3[i] = 1.0 / (1.0 + exp(-tmp));
        }
        
        int index = 0;
        double maxValue = -10;
        
        for (int i = 0; i < level3; i++){
            if (a3[i] > maxValue){
                maxValue = a3[i];
                index = i+1;
            }
        }
        
        out << id << "," << index << "\n";
        while(in.peek() == '\n')
            in.get();
    }
    in.close();
    out.close();
    cout << "predict complete" << endl;
}

void predict1(){
    
    ifstream in("test1.csv"); //读取测试文件
    ofstream out("result1.csv"); //结果文件
    if (!in.is_open()){
        cout << "open file test1"  << " failed" <<endl;
        exit(-1);
    }
    if (!out.is_open()){
        cout << "open result file failed" << endl;
        exit(-1);
    }
    
    cout << "predicting1 ..." << endl;
    out << "id,label" << "\n";
    
    string feature;
    getline(in, feature); //去掉第一行的标签
    
    int id = 0;
    char dot = ' ';
    int label = 0;
    int error = 0;
    int numtest = 0;
    while(1){
        double features[numOfFeatures];
        numtest ++;
        in >> id;
        if(in.eof())
            break;
        //读取617个属性值
        features[0] = 1;
        for (int i = 1; i < numOfFeatures; i++){
            in >> dot;
            in >> features[i];
            //cout << features[i] << endl;
        }
        in >> dot;
        in >> label;
        //计算第二层各节点的值
        a2[0] = 1;
        for (int i = 1;  i < level2; i ++){
            double tmp = 0;
            for (int j = 0; j < numOfFeatures; j++){
                tmp += (theta1[i][j] * features[j]);
            }
            a2[i] = 1.0 / (1.0 + exp(-tmp));
        }
        //计算第三层各节点的值
        for (int i = 0; i < level3; i++){
            double tmp = 0;
            for (int j = 0; j < level2; j++){
                tmp += (theta2[i][j] * a2[j]);
            }
            //cout << tmp << endl;
            a3[i] = 1.0 / (1.0 + exp(-tmp));
        }
        
        int index = 0;
        double maxValue = -10.0;
        
        for (int i = 0; i < level3; i++){
            if (a3[i] > maxValue){
                maxValue = a3[i];
                index = i+1;
            }
        }
        if (label == index)
            error++;
        out << id << "," << index << "\n";
        while(in.peek() == '\n')
            in.get();
    }
    in.close();
    out.close();
    cout << "correct rate = " << error * 1.0 / numtest << endl;
    cout << "predict1 complete" << endl;
}



//把训练出来的theta写入文件
void writeToFile(){
    ofstream out("theta.csv");
    for (int i = 0; i < level2; i++){
        for (int j = 0; j < numOfFeatures; j++){
            out << theta1[i][j] << " ";
        }
        out << "\n";
    }
    
    for (int i = 0; i < level3; i++){
        for (int j = 0; j < level2; j++){
            out << theta2[i][j] << " ";
        }
        out << "\n";
    }
    
    out.close();
    
}

int main(int argc, char *argv[]){
    
    string trainfileName = "train1.csv";  //训练文件
    
    initialize();
    readTrainSamples(trainfileName);
    
    training();
    writeToFile();
    predict();
    predict1();
    return 0;
}
