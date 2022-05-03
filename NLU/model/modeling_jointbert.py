import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF


class IntentClassifier(nn.Module):
    def __init__(self, dim_in, intent_labels_number, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(dim_in, intent_labels_number)

    def forward(self, x):
        return self.linear(self.dropout(x))


class SlotClassifier(nn.Module):
    def __init__(self, dim_in, slot_labels_number, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.linear_layer = nn.Linear(dim_in, slot_labels_number)

    def forward(self, x):
        return self.linear_layer(self.dropout_layer(x))


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.slot_loss_weight = args.slot_loss_coef
        self.intent_labels_number, self.slot_labels_number = len(intent_label_lst), len(slot_label_lst)
        self.pretrained_model = BertModel(config=config)

        self.intent_classifier, self.slot_classifier = IntentClassifier(config.hidden_size, self.intent_labels_number, args.dropout_rate), \
                                                       SlotClassifier(config.hidden_size, self.slot_labels_number, args.dropout_rate)

        self.crf = CRF(num_tags=self.slot_labels_number, batch_first=True) if args.use_crf else None

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.pretrained_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        intent_logits = self.intent_classifier(outputs[1])
        slot_logits = self.slot_classifier(outputs[0])
        
        CE_loss_intent = nn.CrossEntropyLoss()
        intent_loss = CE_loss_intent(intent_logits.reshape(-1, self.intent_labels_number), intent_label_ids.reshape(-1))

        if self.crf is not None:
            slot_loss = -1 * self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
        else:
            logits = slot_logits.reshape(-1, self.slot_labels_number)[attention_mask.view(-1) == 1]
            labels = slot_labels_ids.reshape(-1)[attention_mask.view(-1) == 1]
            CE_loss_slot_filling = nn.CrossEntropyLoss(ignore_index=0)
            slot_loss = CE_loss_slot_filling(logits, labels)
                
        loss = intent_loss + self.slot_loss_weight * slot_loss

        return (loss, (intent_logits, slot_logits),) + outputs[2:]
