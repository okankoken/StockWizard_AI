# StockWizard_AI

BIST (Borsa İstanbul) hisseleri için **gerçekleşmiş hareket** analizleri, **Streamlit paneli** ve **chat ajanı**.  
> **Uyarı:** Bu proje yatırım tavsiyesi değildir.

---

## İçerik

- [Özellikler](#özellikler)
- [Proje Ağacı](#proje-ağacı)
- [Kurulum](#kurulum)
- [Ortam Değişkenleri](#ortam-değişkenleri)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Veri Akışı](#veri-akışı)
- [Komutlar](#komutlar)
- [Sorun Giderme](#sorun-giderme)
- [Lisans](#lisans)

---

## Özellikler

- **Dashboard (Streamlit)**
  - Zaman pencereleri: **1d / 1w / 1m / 3m / 6m / 1y**
  - **Top gainers/losers** sıralama
  - **Hacim anomali** (30g **volume z-score**)
  - Sembol detayında kapanış ve yaklaşık 1A getiri grafiği

- **Chat**
  - Doğal dil: `bugün en çok düşen 5`, `THYAO.IS 1y`, `THYAO.IS haber`
  - Haber araması (opsiyonel Google CSE)

- **Veri Toplama**
  - **Yahoo Finance Chart JSON** → `requests` ile doğrudan çekim (**yfinance yok**)
  - **Incremental upsert** + **cursor**: kaldığı yerden sürer
  - **Exponential backoff** ile 429 rate-limit toleransı
  - TXT’te olup Yahoo’da olmayan sembolleri **skip (not found)**

- **Hesaplamalar**
  - Getiriler: **r_1d, r_1w, r_1m, r_3m, r_6m, r_1y**
  - **vol_z**: 30 günlük hacim z-skoru

---

## Proje Ağacı

```plaintext
StockWizard_AI/
├─ data/
│  ├─ movers.db                # SQLite veritabanı (otomatik oluşur)
│  └─ symbols_bist100.txt      # Sembol listesi (satır başı bir sembol; .IS eklenir)
├─ fetch/
│  ├─ prices.py                # Yahoo chart JSON ile fiyat çekici (incremental)
│  └─ compute.py               # Getiri & hacim z-skoru hesaplayıcı
├─ ui/
│  ├─ app.py                   # Streamlit UI (Dashboard + Chat)
│  └─ news.py                  # Haber arama (opsiyonel, Google CSE)
├─ requirements.txt
├─ run.sh                      # Örnek çalıştırma komutları
└─ setup.sh                    # Örnek kurulum adımları
