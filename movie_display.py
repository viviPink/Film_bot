from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext
import requests
import numpy as np
import random

# Пример списка фильмов 
movies = [
    {"id": 4367, "Title": "The Hunger Games", "poster": "https://occ-0-4922-768.1.nflxso.net/dnm/api/v6/E8vDc_W8CLv7-yMQu8KMEC7Rrr8/AAAABfGM3Z8O6-KvYGD_v_4AxZJT9v2lgwK5sZpTU6gp2DjOmnzFsQ9FNqib9PUAlZCLRPt9TVFhsdi-VI1dYsN0_FbS0qpc.jpg"},
    {"id": 9436, "Title": "Harry Potter and the Philosopher's Stone", "poster": "https://static.kinoafisha.info/upload/news/308532551583.jpg"},
    {"id": 6355, "Title": "Iron Man", "poster": "https://bogatyr.club/uploads/posts/2023-03/1677896581_bogatyr-club-p-toni-stark-foni-vkontakte-14.jpg"},
    {"id": 6359, "Title": "Kung Fu Panda", "poster": "https://bogatyr.club/uploads/posts/2023-03/1679074365_bogatyr-club-p-panda-kunfu-fon-foni-oboi-5.jpg"},
    {"id": 4809, "Title": "Ocean's Eleven", "poster": "https://vistapointe.net/images/oceans-eleven-3.jpg"},
]

liked_movies = []
disliked_movies = []
shown_movies = set()

async def start_recommendation(update: Update, context: CallbackContext) -> None:
    context.user_data['shown_movies'] = set()
    context.user_data['movies_to_show'] = random.sample(movies, min(len(movies), 3))
    await send_movie(update, context)

def recommend_movies(update, context, db, df, embedding_maker) -> None:
    # get_user_feedback_movies(update,df, movies_id)
    a = get_similar_movies(liked_movies, df, db, embedding_maker)
    # result = get_similar_movies_by_info(movie_infos, db, 5)  # Вызываем функцию для поиска похожих фильмов
    return a 

def get_similar_movies(liked_movies, df, db, embeddings_maker, top: int = 10):
    """
    Получить список ближайших фильмов для каждого из лайкнутых фильмов, отсортированный по мере близости.

    @param liked_movies: список названий фильмов, которые пользователь лайкнул
    @param df: DataFrame с исходными данными, включая столбец 'cleaned_plot'
    @param db: объект векторной БД
    @param embeddings_maker: языковая модель для создания эмбеддингов
    @param top: количество ближайших фильмов для каждого лайкнутого фильма
    @return: список ближайших фильмов для каждого лайкнутого фильма, отсортированный по мере близости
    """
    # Список для хранения результатов
    similar_movies = []

    # Для каждого лайкнутого фильма получаем его сюжет и вычисляем эмбеддинг
    for movie in liked_movies:
        plot = df.loc[df['Title'].str.lower() == movie.lower(), 'cleaned_plot']
        if not plot.empty:
            plot = plot.values[0]
            embedding = embeddings_maker.embed_query(plot)

            # Ищем ближайшие фильмы к текущему фильму по его эмбеддингу
            context = db.similarity_search_with_score(plot, k=top)

            # Добавляем результаты поиска в список similar_movies
            for doc, score in context:
                movie_metadata = {
                    'Title': doc.metadata.get('Title', 'Unknown'),
                    'Genre': doc.metadata.get('Genre', 'Unknown'),
                    'Year': doc.metadata.get('Release Year', 'Unknown'),
                    'Director': doc.metadata.get('Director', 'Unknown'),
                    'Score': score
                }
                similar_movies.append(movie_metadata)

    # Сортируем список similar_movies по возрастанию меры близости (Score)
    similar_movies = sorted(similar_movies, key=lambda x: x['Score'])

    return similar_movies

async def send_movie(update: Update, context: CallbackContext) -> None:
    if 'movies_to_show' not in context.user_data or not context.user_data['movies_to_show']:
        await update.message.reply_text("Фильмы закончились!")
        return

    current_movie = context.user_data['movies_to_show'].pop(0)
    context.user_data['current_movie'] = current_movie
    poster = current_movie['poster']
    title = current_movie['Title']

    buttons = [
        [InlineKeyboardButton("👍 Лайк", callback_data='like')],
        [InlineKeyboardButton("👎 Дизлайк", callback_data='dislike')],
    ]
    reply_markup = InlineKeyboardMarkup(buttons)

    if update.message:
        await update.message.reply_photo(poster, caption=title, reply_markup=reply_markup)
    elif update.callback_query:
        await update.callback_query.message.reply_photo(poster, caption=title, reply_markup=reply_markup)

async def like_movie(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    movie_id = context.user_data['current_movie']['id']
    if movie_id not in liked_movies:
        liked_movies.append(movie_id)
    await query.answer()
    await send_movie(query, context)

async def dislike_movie(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    movie_id = context.user_data['current_movie']['id']
    if movie_id not in disliked_movies:
        disliked_movies.append(movie_id)
    await query.answer()
    await send_movie(query, context)