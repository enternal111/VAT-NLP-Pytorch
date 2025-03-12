# VAT-NLP-Pytorch
VAT（虚拟对抗训练）在NLP任务上的Pytorch实现  
训练代码如下：  
vat = VATLoss(model, eps = 2)  
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  
loss = outputs.loss  
vat_loss = vat(attention_mask, outputs.logits)  
total_vat_loss += vat_loss.item()  
loss += vat_loss  
loss.backward()
