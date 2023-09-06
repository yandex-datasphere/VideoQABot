# Создаём вопрос-ответного чат-бота по видео

В последнее время, с появлением больших языковых разговорных моделей (*Large Language Models, LLM*), таких, как [Yandex GPT](https://cloud.yandex.ru/services/yandexgpt), актуальным становится вопрос создания предметно-ориентированных чат-ботов, т.е. таких разговорных чат-ботов, которые способны поддерживать беседу в рамках какой-то узкой предметной области. Такие чат-боты могут быть реализованы двумя путями:

* **До-обучение разговорной модели** - это, как правило, требует значительных вычислительных мощностей, усилий и опыта, и при этом любые изменения в предметной области требуют пере-обучения модели
* **Retrieval-Augmented Generation** - подход, при котором ответ чат-бота формируется стандартной предобученной LLM-моделью, но предварительно ей показывают фрагменты текста из предметно-ориентированной базы знаний, найденные с помощью семантического поиска.

В рамках данного мастер-класса мы создадим вопрос-ответного чат-бота с помощью подхода Retrieval-Augmented Generation, на основе набора видео-файлов. Мы используем [Yandex SpeechKit](https://cloud.yandex.ru/services/speechkit) для преобразования звуковой дорожки видео в текстовый корпус, после чего организуем векторное хранилище и индексацию с помощью [текстовых эмбеддингов Yandex GPT](https://cloud.yandex.ru/docs/yandexgpt/api-ref/Embeddings/) и фреймворка [LangChain](https://www.langchain.com/). В заключении, мы реализуем телеграм-бота, который способен отвечать на текстовые и голосовые запросы.

Подробные комментарии содержатся в файлах проекта, которые рекомендуется открыть в [Yandex DataSphere](https://cloud.yandex.ru/services/datasphere).

![Скриншот работающего бота](images/scrshot.png)

## Этапы работы

Вся работа состоит из следующих основных этапов:

1. [PrepareDataset.ipynb](PrepareDataset.ipynb) - подготовка текстов для поиска из семейства видео. На вход мы получаем набор ссылок на видео на YouTube, скачиваются звуковые дорожки, преобразуются в нужный формат и далее вызывается Yandex SpeechKit для преобразования речи в текст.
1. [LangChainQA.ipynb](LangChainQA.ipynb) - основной код для вопрос-ответного бота на основе LangChain. Здесь текстовые материалы разбиваются на фрагменты, индексируются с помощью эмбеддингов и сохраняются в векторную базу данных. Затем по запросу из базы данных извлекаются релевантные документы, и подаются на вход модели Yandex GPT. В этом же ноутбуке мы разрабатываем простые адаптеры LangChain для Yandex GPT Embeddings и Yandex GPT LLM.
1. [telegram.py](telegram.py) - код телеграм-бота на основе фреймворка flask. Этот скрипт размещается на виртуальной машине.

Чтобы начать работу, вам необходимо проделать следующие подготовительные операции:

1. Получить доступ к Яндекс-облаку - например, в рамках [пробного периода](https://cloud.yandex.ru/docs/free-trial/)
1. Создать в облаке объектное хранилище s3 (в рамках мастер-класса используется имя `s3store`)
1. Создать в облаке [сервисный аккаунт](https://cloud.yandex.ru/docs/iam/concepts/users/service-accounts), имеющий доступ к SpeechKit, YandexGPT и объектному хранилищу, а затем создать [API-ключ](https://cloud.yandex.ru/docs/iam/concepts/authorization/api-key) ([инструкция](https://cloud.yandex.ru/docs/iam/operations/api-key/create)), и параметры этого ключа прописать в файле [config.json](config.json). Также потребуется [создать статический ключ доступа](https://cloud.yandex.ru/docs/iam/operations/sa/create-access-key).

## Пошаговая инструкция

Данный мастер-класс рекомендуется проводить в [Yandex DataSphere](https://cloud.yandex.ru/services/datasphere).

### Подготовка окружения
1. Необходимо создать сообщество в [Yandex DataSphere](https://cloud.yandex.ru/services/datasphere) и проект в этом сообществе.
> Если вы проходите мастер-класс в рамках мероприятия (например, на [Practical ML Conference](https://pmlconf.yandex.ru/)), то вам может быть предоставлен доступ к уже сконфигурированному проекту в DataSphere. 
1. В рамках проекта подключить объектное хранилище `s3store` с помощью созданного статического ключа доступа.
> В рамках мероприятия коннектор к `s3store` может уже быть настроен в рамках сообщества, вам необходимо лишь добавить его в проект.
1. Добавить в проект или создать в проекте секрет `api_key`, содержащий созданный ранее API-ключ к сервисному аккаунту. По этому ключу мы будем вызывать сервисы Speech Kit и Yandex GPT.
1. Изменить активный Docker-образ в проекте на Python 3.10
1. Открыть проект в Jupyter Lab. Рекомендуем использовать режим **Dedicated**. Для режима Serverless, возможно, придётся немного модифицировать пути к хранилищу s3.
1. В разделе GitHub клонировать репозиторий с материалами [http://github.com/yandex-datasphere/VideoQABot](http://github.com/yandex-datasphere/VideoQABot)

### Извлечение текста из видео
1. Откройте в проекте ноутбук [PrepareDataset.ipynb](PrepareDataset.ipynb) и выполните все ячейки кода, обращая внимание на сам код и не инструкции.
1. В качестве источника данных вам потребуются ссылки на несколько видео на YouTube.
1. Вначале аудиодорожки к выбранным роликам скачиваются с помощью библиотеки `pytube` и помещаются в директорию `audio` проекта
1. Поскольку Speech Kit требует определённый формат аудио и частоту дискретизации, с помощью библиотеки `librosa` происходит преобразование аудио к требуемому формату
1. Для преобразования длинного аудио в текст используем [асинхронное распознавание речи](https://cloud.yandex.ru/docs/speechkit/stt/transcribation). В этом случае мы сначала размещаем данные в хранилище s3, и затем запускаем процесс распознавания с помощью REST-запросов. Далее мы в цикле проверяем готовность результатов, и сохраняем их в хранилище.

### Retrieval-Augmented Generation
1. Откройте в проекте ноутбук [LangChainQA.ipynb](LangChainQA.ipynb) и выполните все ячейки кода, обращая внимание на сам код и не инструкции.
1. Для начала документ разбивается на небольшие фрагменты размером `chunk_size`. Размер `chunk_size` нужно выбирать исходя из нескольких показателей:
    * Допустимая длина контекста для эмбеддинг-модели. Yandex GPT Embeddings допускают 2048 токенов, в то время как многие открытые модели HuggingFace имеют длину контекста 512-1024 токена
    * Допустимый размер окна контекста большой языковой модели. Если мы хотим использовать в запросе top 3 результатов поиска, то 3*chunk_size+prompt_size+response_size должно не превышать длины контекста модели.
1. Далее мы учимся считать по фрагментам текста векторные эмбеддинги, с помощью моделей от HuggingFace, или через Yandex GPT Embedding API. В последнем случае нам пришлось написать адаптер для LangChain для работы с Yandex GPT Embeddings.
1. Создаём векторную базу данных, обрабатываем все фрагменты и сохраняем их
1. Учимся извлекать релевантные фрагменты по запросу
1. Пишем адаптер для LangChain для работы с моделью Yandex GPT. Убеждаемся, что Yandex GPT работает, но не очень хорошо отвечает на предметно-ориентированные запросы.
1. Собираем цепочку для Retrieval-Augmented Generation и проверяем её работу

### Создаём вопрос-ответного бота в телеграм
Для создания вопрос-ответного бота нам потребуется развернуть нашу цепочку LangChain в виде публично-доступного веб-сервиса по HTTPS. Это удобнее всего сделать с помощью виртуальной машины Yandex Compute. Для понимания того, как устроены боты в телеграм, можно порекомендовать [эту документацию](https://core.telegram.org/bots/tutorial).

1. Создаём виртуальную машину. Для экспериментов нам не нужна высокая производительность, будет достаточно 4-6 Gb RAM, 50 Gb SSD, Ubuntu. Для входа на виртуальную машину используется ssh-сертификат.
> Код телеграм-бота подразумевает, что пользователь на виртуальной машине будем иметь имя `vmuser`. Если вы используете другое имя, то придётся внести исправления в код.
1. Создаём для виртуальной машины статический IP-адрес
1. Для работы с телеграм потребуется HTTPS-протокол и сертификат SSL. Поэтому необходимо привязать к виртуальной машине какое-то доменное имя.
1. Заходим в консоль виртуальной машины по SSH
1. Клонируем репозиторий проекта `git clone https://github.com/yandex-datasphere/VideoQABot`
1. Переходим в каталог проекта и устанавливаем зависимости:
```
cd VideoQABot
pip3 install -r requirements.txt
```
1. Создаём SSL-сертификат для выбранного ранее доменного имени, это можно сделать, например, с помощью бесплатного сервиса *Let's Encrypt* и `certbot`
1. Сертификаты записываем в директорию `cert`, и прописываем путь к ним в файле [`telegram.py`](telegram.py) 
1. Создаём бота в телеграм при помощи `botfather` (см. [док](https://core.telegram.org/bots/tutorial#getting-ready)), и полученный telegram token записываем в [`config.json`](config.json)
1. Также в [`config.json`](config.json) прописываем адрес нашего сайта. Рекомендуется использовать порт 8443, поскольку в этом случае запускать веб-сервер можно от имени обычного пользователя.
1. Копируем векторную базу данных, полученную на предыдущем шаге, в директорию `store`.
1. Запускаем `python3 telegram.py`

На этом этапе вы должны быть в состоянии послать в бота сообщения, текстом или как голосовое сообщение, и получить ответ, текстом + голосом.
