
# Joint Entity Extraction and Assertion Detection for Clinical Text

**From ACL 2019** [paper](https://arxiv.org/pdf/1812.05270.pdf)

----
### 原始数据schame
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

### 处理后数据schame
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

----
## model test
```
python model.py
```

