### 前言
這是到 Wehelp 訓練營第二階段第五週為止我在 ptt 文章分類器的訓練流程上的一些紀錄，由於自己所實驗的方向與得到的數據我認為不一定與繳交作業所要提供的數據相吻合，所以請容我用我這個獨特的脈絡描述我的實驗歷程，以及過程中所得到的數據、啟發與抉擇，過程中也會引述到一些數據的記錄檔以供參考。那麼，就開始這趟旅程吧～

### Cleaning & Tokenize & Embedding 
從執行 embedding model 的訓練開始，我就遭遇了很大的困難，我一直無法突破彭彭所訂立的 `self_similarity >= 0.7 / second_similarity > 0.8` 這個基礎要求，這件事讓我感到十分焦慮，主要原因為看到 demo 呈現時發現彭彭可以跑出 `self_similarity ~= 0.8 / second_similarity ~= 0.9` 這個數據，但我始終在 `self_similarity ~= 0.6X / second_similarity ~= 0.7X` 這個水平徘徊，但這並不是我很懶，隨便試了幾個組合就放棄，實際上我總共嘗試了 1000 多個組合([檔案連結](./data/ex/embedding_result.csv))，並從不同的組合中逐漸收納訓練的參數，才在最後擠出了接近「基礎要求」的數據。我得出的結論大概是這樣的：
- `epochs`: 大約 10 次左右是個甜蜜點，更小或更多都會使 similarity 下降，至於為什麼會這樣，至今我仍然想不通，理論上訓練越多次頂多是產生邊際效應讓 `self_similarity` 到達一個巔峯值使得每個 epochs 能增進的程度變得越來越小，但以我的實驗數據來看，明顯不是這樣的，付出多次努力反而使模型有了些偏誤。
- `window`: 2 是後來實驗出大概「最好」或「足夠」的參數，我認為這和訓練文本「標題」這樣的短文性質有關，標題的本質就是要在很少的字數下傳達精確的概念，相隔太遠的文字通常關聯性會低到不可思議，再多幾個文字一起考慮也不會對這種類型的文本 embedding 有更好的效果。
- `min_count`: 由於我們的文本量還算多（有100多萬筆），但同時他們又限定於特定領域，文字本身就會頻繁地重複，以我自己對數據的觀察是在這種情況下只要不設 1，可能對於訓練結果都差不多。我自己後來統一用 5，這個數字沒什麼特別原因，就只是它在我初期的實驗數據中剛好都有拿到不錯的結果，但也是有類似 3, 10 等數據也能呈現差不多水平的結果，總之我認為它在這次的任務不是特別重要。
- `training algorithms`: 我一直是使用預設的 `PV-DM`，中間依稀曾有切為 `PV-DBOW` 嘗試效果，但初步看起來沒有太多差距，最後我就沒有將他視為一個變數記錄在 `embedding_result.csv` 中
- `hs / negative / sample`: 這三個變數我最後沒有細究他們的核心原理，以我的實驗結果來說，設為 1 / 0 / 0 大概是一個可以穩定輸出的數字。我認為這些組合在這個任務上不是調教的重點，即使有調整差異也不大，就放著。
- `vector_size`: 以提高 `similarity` 這個任務而言，50 即可達到最高可辨識度的水平，再往上似乎也會讓模型感受到混淆。
- `worker`: 以我使用的 arm 架構 MacBook 為例，無論我開的 worker 是 4, 5, 8, 10。不知為何最多都只能拿出電腦一半的 cpu 進行訓練(電腦總共有 10 個 cpu，另外 本 embedding model 只能用 cpu 訓練不能以 gpu 加速，所以純看 cpu 核心數即可)，由於更多的 worker 只會讓他們互相搶食 cpu 資源反而讓運算更慢，最後我的寫法就是拿到 cpu 數量後 // 2 = worker 數量 (=5)。

但如上述的結論，這樣努力實驗的數據在 similarity 測試只能過基本門檻，且在後續訓練 title_classifier 一樣卡在 6X％ 的正確率，於是我便從更前面的步驟：tokenizer 開始思考是不是有哪裡做錯了？
後來我在檢查 [tokenized_file](./data/ex/tokenized_file.txt.sample) 後，發現了一個重要因子：我將每個標題中的「文章分類」(ex. 討論、情報、新聞、閒聊)也放入了 tokenized_file 中，這用膝蓋想就知道肯定是個混淆字詞，畢竟每個版的每個文章都很有可能出現這些分類，這些文字還都被放在第一個，也增加了文章之間的相似性，模型在比對時自然就會因為這個多餘的文字而出錯，同時他也有 87% 對於後續 title_classifier 的分類毫無幫助，於是大工程就產生了，我要將[文章清理](./cleaner.py)與其後續的流程 (tokenizer / embedding) 重新走一遍！
最後文章清理 (cleaning) 的規則如下：
1. 將文章的前後空白刪除，並將英文全轉小寫
2. 將 re: / fw: 刪除
3. 將開頭 `[xxx]` 整段清除，因為他通常是文章分類，就算有機率誤刪也沒關係，錯殺一個總比放過一百個要好。
Tokenize 的規則則是：
1. 分詞後依據詞性決定是否要留下來
2. 留下來的分詞也要再將前後空白刪除，減少不必要資訊。
爬蟲爬取的文章數: 1,067,546
tokenized_file 文章數：1,067,546

可以看到 [data](./data/) dir 有 [ex/](./data/ex/) dir，其中存放的都是 cleaner 改版前的數據。
改版後的 embedding 實驗結果與數據可參照[此檔](./data/embedding_result.csv)

#### 4/14 數據更新
- `dm=0 / dbow_words=1`: 適合短文的組合，即使 similarity 的測試並沒有特別突出，但在後續的 title classifier 能建立神奇的功效，將 test accuracy 一股從 65% 左右的門檻拉到 84% ，非常神奇！！！
最終 embedding model vector_size: 30


### Title Classifier
到了最關鍵的分類模型，我設計了一個架構幫助我快速閱覽不同參數的實驗數據，
實驗數據記錄檔：命名規則 `title_classification_result_{data_size}.csv`，幫助我快速評估不同訓練資料量的呈現結果，裡面會記錄`模型樣態`、`累積 epochs` 與`預測正確率`。
訓練模型：命名規則 `title_classifier_{layer_sizes}_{data_size}.pth`，讓我快速瀏覽有哪些 layer 組合已經被嘗試過。
一開始的 [5000 筆資料](./data/title_classification_result_5000.csv) 為亂槍打鳥的嘗試，跑一組 epoch 不到 1 秒，可以快速查看 epoch 數量對預測正確率的進展程度，但才 5000 筆資料沒什麼訓練價值可言，後來跳到 [100,000 筆資料](./data/title_classification_result_100000.csv)，實驗才開始有意義，實驗大致歷程如下：
1. 實驗 hidden layer 數量對模型的影響 -> 發現 hidden layer 兩層的效果較好，中間順便測試 SGD & Adam 發現，Adam 能快速把正確率在很少的 epoch 就拉到相對的水平(60%)
2. 以固定輪次 (50 epochs) 並以「後一個 layer vector_size ~= 前一個 layer vector_size / 2 的遞減方式」實驗 `input_layer` (文章 embedding) `vector_size` 對模型的影響 -> 發現 `vector_size` 越高似乎效果越好。
3. 由於 embedding layer `vector_size` 越大 model size 也越大，尤其 `vector_size` 300 的 embedding layer 已經大到有 1.X G，認為不是好方法，於是嘗試用較小的 input layer `vector_size` 接到較大的 hidden layer `vector_size` 看效果如何，並將 epochs 改為 10 加速實驗 -> 發現效果還不錯。
4. 發現 AI 幫忙改的 code 在 model sequence 上把 SoftMax 加了上去，於是刪掉後再繼續嘗試更大的第一層 hidden layer -> 發現有奇效，直到 1000 左右的 `vector_size` 效果都還在往上飆，但測試資料的準確率大概碰到了上限（約 65% ）。
5. 開始嘗試以 [500,000筆資料](./data/title_classification_result_500000.csv)進行訓練，並選擇 150 `vector_size` embedding model 當作基底（模型大小共 700 mb 還可接受）-> 效果不錯
6. 確認實驗方向正確，以 [1,000,000筆資料](./data/title_classification_result_1000000.csv) 以相同 embedding model 當作基底進行訓練，但增大 epoch 數量以求最大預測率。

最後的各項參數指標
1. Arrangement of Linear Layers: `150*1000*50*9`
2. Activation Function for Hidden Layers: ReLU
3. Activation Function for Output Layers: Softmax
4. Loss Function: Categorical Cross Entropy
5. Algorithms for Back-Propagation: Adam (Adaptive Moment Estimation)
6. Total Number of Training Documents: 800,000
7. Total Number of Testing Documents: 200,000
8. Epochs: 300 / Learning Rate: 0.001
9. Train Accuracy: 78.2% / Test Accuracy: 64.7%


#### 4/14 更新
經過 embedding model 的重新調教後，數據如下
1. Arrangement of Linear Layers: `30*1000*500*9`
2. Activation Function for Hidden Layers: ReLU
3. Activation Function for Output Layers: Softmax
4. Loss Function: Categorical Cross Entropy
5. Algorithms for Back-Propagation: Adam (Adaptive Moment Estimation)
6. Total Number of Training Documents: 800,000
7. Total Number of Testing Documents: 200,000
8. Epochs: 30 / Learning Rate: 0.001
9. Train Accuracy: 93.7% / Test Accuracy: 84%


### 後續思考
1. 以本項任務來說，有個體悟是 embedding 本身在 `similarity` 的成效本身其實並不用看到太重，文章分類確實是個問題可以優化，但其實不需要追求到很高的 `self-similarity`，原因是我們最終又要把這些複雜的 vectors 分類為僅僅 9 個類別，即使他們在個別辨識上有很高的分辨率，對於「分類」這個問題可能也不是很關鍵的因素。
2. 在過去的週一分享就有聽過不只一次有個訓練的方法是在 `hidden_layer` 拉大 `vector_size`，這件事本身很反直覺，我們應該是將變因(`vector_size`)逐漸縮小，就可以找到正確的類別，但最後實驗證實逐漸縮小的成效只能到某個程度就會停下來。而真的去實際嘗試這件事也是因爲發現 `input_layer` 的 `vector_size` 越大，成效就越好，才想到或許可以嘗試在 `hidden_layer` 加大的做法，不要讓 embedding model 無限擴張，沒想到有如此巨大的成效，算是一個蠻有趣的 take away。
3. 最後得到這樣的 Train Accuracy 和 Test Accuracy，我覺得算是有 over fitting 的問題，但因為時間因素，已經沒空再深入探討要怎麼優化了。這趟旅程最大的啟發就是還是要找時間讀書，先累積一些前人研究過的概念（如何訓練等），才能快速在實驗中驗證，自己胡搞摸索就會像這次一樣最後搞不出個厲害的模型。這議題就又有點像一般在公司時做專案的兩難：你是要先用比較醜的方式把功能做出來，還是你要研究出「比較好的寫法」才實作，有時候因為和時間賽跑的緣故也由不得自己選，在平時是否有在累積些東西就變得很重要。
4. 近一步思考這個模型如果做到最好，我猜測在預期新文章的準確率可能也頂多接近 70~80%，原因來自文章標題中所涵蓋的意義會隨著時代快速的變動，我們無法預測到下一季會有哪些動畫，在 c_chat 的標題會變得難以捉摸，我們也不會知道政治情勢三個月後的走向，hatepolitics 在討論的範圍可能截然不同，甚至不是過去所討論的議題或人物。當然也有些版可能討論的內容大同小異(x. boy-girl / pc_shopping)， 5090 顯卡過了一年也就變成 6090，相信模型還是抓得出來，男女議題也是千古不變。只是我們不得不承認，在無法有 agent 搜尋當下資料的模型，就只能像是過去的 chatgpt 一樣，不斷闡述「我的資料只到 2023/XX/XX」，這樣的模型會隨著時間的流逝越來越與當下脫節，最後變得不敷使用。在建造完單純的模型後如何與時間賽跑，目前大概還是個無解的難題。
#### 4/14 更新
5. 找到關鍵參數是訓練歷程中最重要的部分，漏掉關鍵參數即會失去模型寶貴的判斷力，算是學到一課，多讀書、多去嘗試新的方向是極為重要的成功指標，未來在訓練時一定要謹記這份教訓，也非常感謝彭彭邀請大家在會議中分享，訓練結果才能有這樣的巨大改變。