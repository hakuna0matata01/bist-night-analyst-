# BIST Akşam Taraması — Faz 1

## v3 — Teknik Analiz Doğruluğunu Güçlendiren Düzenlemeler

Bağımsız bir kod incelemesinde tespit edilen noktalara karşı yapılan değişiklikler:

| # | Sorun | Çözüm |
|---|-------|-------|
| 1 | RSI/ADX, pandas `ewm(alpha=1/period)` ile **yaklaşık** Wilder smoothing kullanıyordu (kısa dönemlerde ~1 puana varan sapma) | `_wilder_rma()` ile klasik, SMA ile başlatılan **tam** Wilder özyinelemesi uygulandı. Sentetik veriyle test edildi: RSI artık klasik referansla **birebir** (fark = 0.0), ADX farkı 50. bardan sonra <0.2 puan |
| 2 | Direnç seviyesi sadece **kapanış** fiyatlarına dayanıyordu, gün içi fitilleri yok sayıyordu | `recent_high20` artık **High** (günlük en yüksek) fiyatlara dayanıyor — daha standart bir direnç/kırılım tanımı |
| 3 | Momentum kriteri (RSI 40-65) ile Seviye kriteri (direnç kırılımı) çelişiyordu: gerçek bir kırılım genelde RSI'ı 65'in üzerine taşır ve momentum puanını "cezalandırırdı" | Momentum artık iki senaryoyu kapsıyor: erken/orta aşama (RSI 40-65, yükseliyor) **veya** kırılım devamı (Seviye kriteri sağlanmışsa RSI 65-80 arası, yükseliyor). 80 üzeri hâlâ aşırı-alım sayılır |
| 4 | Tarihsel başarı oranı tek hissenin kendi geçmişiyle sınırlıydı (n genelde 5-20, istatistiksel olarak zayıf) | Yeni **"Tüm evren bazlı tarihsel doğrulama"** paneli: taranan tüm hisselerin sinyal-günleri birleştirilip (cross-sectional) çok daha büyük bir örneklemle genel geçerlilik ölçülüyor. Hem tek-hisse hem tüm-evren sonuçlarına **%95 Wilson güven aralığı** eklendi (küçük örneklemde yanıltıcı kesinlik vermez) |
| 5 | Yahoo Finance'in doğru ayarlamadığı bedelsiz sermaye artırımı/bölünme gibi olaylar, tüm göstergeleri (EMA/RSI/ADX) sessizce bozabiliyordu | Yeni `detect_data_quality_issue()`: tek günlük değişim %30'u geçerse (BIST'in fiili ±%10 bandının çok üzerinde) hisse "veri kalitesi sorunu" olarak işaretlenip analiz dışı bırakılıyor, ayrı bir listede gösteriliyor |
| 6 | BIST100 endeksi çekilemezse, Göreli Güç kriteri sessizce devre dışı kalıyor ama arayüzde küçük bir notla geçiştiriliyordu | Artık belirgin bir uyarı (`st.error`) gösteriliyor ve o taramadaki **azami puanın 7 değil 6 olduğu** hem üstte hem her hissenin detay panelinde açıkça belirtiliyor |

Bu değişiklikler puanlama **felsefesini** değiştirmez (hâlâ kural tabanlı bir ön-eleme aracıdır), sadece hesaplamaların standart tanımlarla tam örtüşmesini ve istatistiksel iddiaların (başarı oranı gibi) daha dürüst sunulmasını sağlar.


Piyasa kapandıktan sonra (örn. saat 20:00) çalıştırılan, BIST hisselerini
teknik kriterlere göre puanlayıp ertesi gün için kısa bir **izleme listesi**
(watchlist) çıkaran bir Streamlit uygulaması.

> ⚠️ Bu araç yatırım tavsiyesi değildir. Sadece teknik kriterlere dayalı bir
> ön-eleme/tarama aracıdır.

## Puanlama Mantığı (v3 — maks. 7 puan, risk cezalı; endeks verisi yoksa azami 6)

**Pozitif kriterler (her biri +1):**

| # | Kriter | Açıklama |
|---|--------|----------|
| 1 | Trend | Fiyat > EMA20 > EMA50 (boğa dizilişi — sadece fiyat>EMA değil) |
| 2 | Momentum | RSI(14) 40-65 bandında ve yükseliyor **veya** (Seviye kriteri sağlanmışsa) RSI 65-80 arası ve yükseliyor |
| 3 | MACD | MACD çizgisi sinyalin üzerinde, histogram güçleniyor |
| 4 | Hacim | Günlük hacim 20g ortalamanın **1.3x** üzerinde + fiyat yukarı |
| 5 | Seviye | **Günlük en yüksek (High) fiyatlara dayalı** direnç kırılımı/yakınlığı ya da sıkışma (düşük volatilite) sonrası |
| 6 | ADX | ADX(14) ≥ eşik — klasik Wilder formülüyle (tam, yaklaşık değil) hesaplanır |
| 7 | Göreli Güç | Son 20 günde **BIST100 endeksinden** daha güçlü performans (endeks verisi alınamazsa bu kriter hesaplanamaz, azami puan 6'ya düşer) |

**Risk cezaları (her biri -1, toplam puan 0'ın altına inmez):**

| Bayrak | Açıklama |
|--------|----------|
| Aşırı uzama | Fiyat, EMA20'nin %10+ üzerinde (kovalama/geri çekilme riski) |
| Marj sınırı | Bugünkü değişim zaten ±%9+ (yarın için gap/tepki riski) |

**Ek bağlam ve filtreler:**
- **Piyasa rejimi:** BIST100 endeksinin kendi trendi (EMA50 üzerinde mi) üstte
  bir bilgi/uyarı panosu olarak gösterilir. Endeks zayıfken bireysel sinyaller
  istatistiksel olarak daha az güvenilir olma eğilimindedir.
- **Likidite filtresi:** 20 günlük ortalama TL cirosu, sol menüden
  belirlenen eşiğin altındaki hisseler izleme listesinden (isteğe bağlı,
  varsayılan açık) hariç tutulur — ince/az işlem gören hisselerde teknik
  sinyaller güvenilmezdir ve pratikte alım/satımı da zor olabilir.
- **Tarihsel Başarı Oranı:** Her hissenin **kendi geçmiş verisinde**, aynı
  puan eşiğini geçtiği günlerden sonraki günün pozitif kapanma oranı ve
  ortalama getirisi hesaplanır (vektörize mini-backtest). Yeterli sinyal
  yoksa (<5) "—" gösterilir. **Bu geçmiş performanstır, gelecekteki sonucu
  garanti etmez.**

## Giriş Tetik Seviyesi

Orijinal Faz 1 planındaki *"her biri için giriş tetik seviyesi (dünkü direncin
X kuruş üzeri) not et"* adımı, sonuç tablosunda ayrı sütunlar olarak hesaplanır:

- **Referans Direnç:** O gün için izlenecek direnç — hisse bugün henüz
  direncini kırmadıysa, dünden bugüne taşınan 20 günlük direnç seviyesi; eğer
  bugün zaten kırılım yaptıysa (daha önceki direncin üzerinde kapandıysa),
  bugünün kapanışı yeni referans olur (devamlılık teyidi aranır).
- **Giriş Tetik Fiyatı:** Referans Direnç × (1 + tampon%). Yarın fiyat bu
  seviyeyi yukarı keserse (ve Faz 2'deki RVOL/yön teyidi de sağlanırsa) giriş
  düşünülebilecek fiyat budur.
- **Tetiğe Uzaklık %:** Bugünkü kapanışın tetik fiyatına olan yüzde mesafesi.

**Neden sabit kuruş değil de yüzde tampon?** BIST'te hisseler birkaç TL'den
binlerce TL'ye kadar çok farklı fiyat seviyelerinde işlem görüyor; 634 hissenin
tamamında anlamlı kalması için tampon yüzde olarak uygulanır (sol menüden
ayarlanabilir, varsayılan %0.5). Bu seviye grafik üzerinde turuncu kesikli
çizgi olarak da işaretlenir ve İzleme Listesi CSV'sine dahildir — ertesi sabah
Faz 2'de doğrudan kullanılabilir.

Bu, stratejinin **Faz 2** (sabah 10:30 sonrası Relative Volume teyidi ve
alım) adımından önceki hazırlık aşamasıdır; Faz 2 ayrı bir uygulama/modül
olarak eklenebilir.

## Yerelde Çalıştırma

```bash
git clone <repo-url>
cd bist-screener
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Tarayıcıda otomatik olarak `http://localhost:8501` açılır.

## GitHub'a Yükleme

```bash
cd bist-screener
git init
git add .
git commit -m "BIST akşam taraması - Faz 1"
git branch -M main
git remote add origin https://github.com/KULLANICI_ADIN/bist-screener.git
git push -u origin main
```

## Streamlit Community Cloud'a Dağıtım

1. [share.streamlit.io](https://share.streamlit.io) adresine GitHub hesabınızla giriş yapın.
2. **"New app"** butonuna tıklayın.
3. Az önce push ettiğiniz repoyu, branch'i (`main`) ve ana dosyayı (`app.py`) seçin.
4. **Deploy** butonuna basın. Birkaç dakika içinde uygulamanız canlıda olur
   (örn. `https://bist-screener-xxxx.streamlit.app`).
5. Her `git push` sonrası uygulama otomatik olarak güncellenir.

Her akşam saat 20:00 civarı bu linke girip "Taramayı Başlat"a basmanız yeterli.

## Hisse Listesi

Uygulama üç kaynaktan hisse listesi alabilir (sol menüden seçilir):

- **Tüm BIST Hisseleri** *(varsayılan)* — `data/bist_tickers.txt` dosyasından
  okunur. Bu dosya, yüklenen "BIST-tumhisse.xlsx" (isyatirim) listesinden
  çıkarılan **634 hisse kodunu** içerir, bir satırda bir kod olacak şekilde.
  Listeyi güncellemek için bu `.txt` dosyasını düzenlemeniz yeterli — kod
  değişikliği gerekmez.
- **Örnek alt küme (BIST30)** — hızlı test/deneme için küçük, sabit bir liste.
- **Kendi listem** — virgülle ayrılmış, istediğiniz hisseler (`.IS` uzantısı
  otomatik eklenir).

### Tam liste taramasında performans

634 hissenin tamamını taramak, 30 hisseyi taramaktan çok daha uzun sürer:

- İndirme, Yahoo Finance hız sınırlarını aşmamak için **100'lük gruplar**
  halinde ve paralel (`threads=True`) yapılır (`DOWNLOAD_CHUNK_SIZE` ile
  `app.py` içinden ayarlanabilir).
- Sonuçlar **2 saat** boyunca önbellekte tutulur (`fetch_batch` üzerindeki
  `ttl=7200`); aynı akşam tekrar tekrar basmanıza gerek kalmaz.
- Likit olmayan/az işlem gören bazı kodlarda veri gelmeyebilir; bunlar
  "Analiz edilemeyen hisseler" kısmında listelenir, uygulamayı durdurmaz.
- Daha hızlı yinelemeli test için "Örnek alt küme (BIST30)" seçeneğini
  kullanmanızı, tam taramayı günde bir kez (akşam) çalıştırmanızı öneririz.

## Özelleştirme

- **Arayüzden canlı ayarlanabilenler:** minimum puan eşiği, minimum ADX
  eşiği, minimum ortalama günlük ciro (likidite), düşük likiditeyi hariç
  tutma, geçmiş veri aralığı.
- **Kodda sabit olan eşikler** (`app.py` başındaki sabitler): RSI bandı
  (40-65), kırılım durumunda izin verilen üst RSI sınırı (`RSI_BREAKOUT_UPPER`,
  80), hacim çarpanı (1.3x), direnç toleransı (%2), aşırı uzama eşiği
  (`OVEREXTENSION_PCT`, %10), marj sınırı eşiği (`NEAR_LIMIT_PCT`, %9),
  göreli güç bakış penceresi (20 gün), minimum backtest sinyal sayısı
  (`MIN_BACKTEST_SIGNALS`, 5), minimum tüm-evren backtest sinyal sayısı
  (`MIN_AGGREGATE_BACKTEST_SIGNALS`, 30), şüpheli fiyat sıçraması eşiği
  (`SUSPICIOUS_JUMP_PCT`, %30). Kendi stratejinize göre bu sabitleri
  değiştirebilirsiniz.
- **Endeks referansı:** Göreli güç ve piyasa rejimi notu `INDEX_TICKER =
  "XU100.IS"` (BIST100) üzerinden hesaplanır; isterseniz başka bir endekse
  çevirebilirsiniz.
- **Hisse listesini güncelleme:** isyatirim.com veya BIST'in resmi
  kaynaklarından yeni bir liste alıp `data/bist_tickers.txt` dosyasının
  içeriğini (bir satır = bir hisse kodu, `.IS` olmadan) değiştirerek
  güncelleyebilirsiniz.

## Veri Kaynağı

Veriler [Yahoo Finance](https://finance.yahoo.com) üzerinden `yfinance`
kütüphanesi ile çekilir (BIST hisseleri `.IS` uzantısıyla, örn. `THYAO.IS`).
Yahoo Finance verileri gecikmeli/gayri resmi olabilir; kritik kararlar için
resmi veri kaynaklarıyla teyit edin.

## Olası Sonraki Adımlar

- Faz 2 (sabah RVOL teyidi + VWAP kontrolü) için ayrı bir sayfa/sekme eklemek
- Tarama sonuçlarını otomatik olarak her akşam GitHub Actions ile çalıştırıp
  sonucu bir CSV/Telegram bildirimi olarak almak
- Walk-forward / tüm-liste genelinde toplu backtest (şu an her hissenin
  kendi geçmişiyle sınırlı; sektörler arası veya tüm BIST genelinde
  karşılaştırmalı bir backtest eklenebilir)
- Sektör bazlı gruplama ve sektör göreli gücü (tek tek hisse yerine sektör
  rotasyonunu da dikkate almak)
