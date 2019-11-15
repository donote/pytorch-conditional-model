
**执行前先设置工作路径**
```
export PYTHONPATH=`pwd`
```

----
@time 2019.11.08
- 预处理样本，生成vocab并增加BIO标识
```bash
# 生成vocab，所在目录为配置中data_root
# 此处为: data/ner_status/data/
python model_v2/models/ner_status/main.py --exp_name symptom --mode prep
```

- 模型训练
```bash
# 配置config.py和symptom.json
python model_v2/models/ner_status/main.py --exp_name symptom --mode train
```

----
@time 2019.11.08

- 对Hsin中病历进行句子拆分，然后根据symptom词表进行简单的预标
```bash
python pyscript/symptom_gen.py > data/ner_status/raw_data/symptom_oc_ann.txt
```

- 对训练样本进行train/test/dev划分，默认划分比例为8:1:1, 划分后的数据和input文件是一个目录
```bash
python model_v2/utils/sample_split.py -i data/ner_status/raw_data/symptom_oc_ann.txt
```


----
@time 2019.10.25
### 增加ner对应的状态联合学习模型

### 数据处理思路

- 原始数据按句号「。」进行划分，单个句子作为样本，这一步可以独立与模型处理，暂时先不考虑
- 单句子样本数据格式如下：

```json
{
'text': '患者目前咳嗽咳痰，偶有头晕，无头痛，胸闷气促，胃寒发热等不适。',
'urid': 'none',
'weizhi': [
		{'label': u'症状'， data': '头晕', 'pos': [start, end], 'status': 0}, 
		{'label': u'症状'，'data': '头痛', 'pos': [start, end], 'status': 1},
		{'label': u'症状'，'data': '胸闷气促', 'pos': [start, end], 'status': 1},
		{'label': u'症状'，'data': '胃寒发热', 'pos': [start, end], 'status': 1}
        ]
}
```
- 处理为用于训练的样本
```json
{
"text": "给予患者阿帕替尼片靶向治疗", 
"urid": "none", 
"sid": 5, 
"index": [40, 53], 
"anns": [{"start": 4, "end": 9, "value": "阿帕替尼片"}],
"bio": "O", "O", "O", "O", "B", "I", "I", "I", "I", "O", "O", "O", "O"],
"status": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
}

```
- 从上面格式中的bio和status看，可以将status和bio合并在一块，构建新的BIO标识，正向极性NER_1, 负向极性NER_0

- 使用统一的vocab

- 在联合训练中着重修改评估方式，一是实体全匹配评估，二是极性在实体上的二分类


