from torch import mean, ge, no_grad, sigmoid, cat

def cuda(x):
    return x.cuda(non_blocking=True)

def f1_score(y_true, y_pred, threshold=0.5):
    return fbeta_score(y_true, y_pred, 1, threshold)


def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))
    
    
    


def train_one_epoch(model, train_loader, criterion, optimizer, steps_upd_logging = 250):
    model.train();
    
    total_loss = 0.0
    loader = iter(train_loader) 
    step = 0
    for features, targets in loader:
        step += 1
        features, targets = cuda(features), cuda(targets)
        optimizer.zero_grad()
        
        logits = model(features)
        
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (step + 1) % steps_upd_logging == 0:
            logstr = f'Train loss on step {step + 1} was {round(total_loss / (step + 1), 5)}'
            #train_tqdm.set_description(logstr)
            print(logstr)
        
    return total_loss / (step + 1)


def validate(model, valid_loader, criterion):
    model.eval();
    test_loss = 0.0
    true_ans_list = []
    preds_cat = []
    step = 0
    with no_grad():
        step += 1
        valid_iterator = iter(valid_loader)
        
        for features, targets in valid_iterator:
            features, targets = cuda(features), cuda(targets)

            logits = model(features)
            loss = criterion(logits, targets)

            test_loss += loss.item()
            true_ans_list.append(targets)
            preds_cat.append(sigmoid(logits))

        all_true_ans = cat(true_ans_list)
        all_preds = cat(preds_cat)
                
        f1_eval = f1_score(all_true_ans, all_preds).item()

    logstr = f'Mean val f1: {round(f1_eval, 5)}'
    print(logstr)
    return test_loss / (step + 1), f1_eval
