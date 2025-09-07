import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objs as go
import plotly.io as pio
import pycountry


class EDA:
    def __init__(self, dfs):
        self.df = dfs["df"]
        self.credits_df = dfs["credits_df"]
        self.keywords_df = dfs["keywords_df"]
        self.links_df = dfs["links_df"]
        self.ratings_df = dfs["ratings_df"]
        self.merged_df = dfs["merged_df"]
        self.img_path = "D:/Uni/Term 6/Machine Learning/HomeWork/6/report/images/"
        os.makedirs(self.img_path, exist_ok=True)

    def plot_rating_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.merged_df['rating'], bins=10, kde=False)
        plt.title('Distribution of Movie Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.img_path, "rating_distribution.png"), bbox_inches='tight')
        plt.close()

    def plot_release_year_distribution(self):
        df = self.merged_df.copy()
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        plt.figure(figsize=(12, 6))
        sns.histplot(df['release_year'].dropna(), bins=50, kde=False)
        plt.title('Distribution of Movie Release Years')
        plt.xlabel('Release Year')
        plt.ylabel('Number of Movies')
        plt.savefig(os.path.join(self.img_path, "release_year_distribution.png"), bbox_inches='tight')
        plt.close()

    def plot_budget_vs_revenue(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.merged_df, x='budget', y='revenue')
        plt.title('Relationship between Movie Budget and Revenue')
        plt.xlabel('Budget')
        plt.ylabel('Revenue')
        plt.savefig(os.path.join(self.img_path, "budget_vs_revenue.png"), bbox_inches='tight')
        plt.close()

<<<<<<< HEAD
        # Convert 'budget' and 'revenue' to numeric, coercing errors to NaN
        self.merged_df['budget'] = pd.to_numeric(self.merged_df['budget'], errors='coerce')
        self.merged_df['revenue'] = pd.to_numeric(self.merged_df['revenue'], errors='coerce')

        # Fill NaN values in 'budget' and 'revenue' with 0, as 0 budget/revenue is a meaningful value
        self.merged_df['budget'] = self.merged_df['budget'].fillna(0)
        self.merged_df['revenue'] = self.merged_df['revenue'].fillna(0)

        # Filter out movies with zero budget AND zero revenue
=======
        self.merged_df['budget'] = pd.to_numeric(self.merged_df['budget'], errors='coerce')
        self.merged_df['revenue'] = pd.to_numeric(self.merged_df['revenue'], errors='coerce')

        self.merged_df['budget'] = self.merged_df['budget'].fillna(0)
        self.merged_df['revenue'] = self.merged_df['revenue'].fillna(0)

>>>>>>> 848b2a5 (almost finilaize)
        filtered_df = self.merged_df[(self.merged_df['budget'] > 0) | (self.merged_df['revenue'] > 0)].copy()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=filtered_df, x='budget', y='revenue')
        plt.title('Relationship between Movie Budget and Revenue (Filtered)')
        plt.xlabel('Budget')
        plt.ylabel('Revenue')
        plt.savefig(os.path.join(self.img_path, "budget_vs_revenue_filtered.png"), bbox_inches='tight')
        plt.close()

    def plot_genre_counts(self):
        genre_counts = {}
        for genres_list in self.df['genres'].dropna():
            if isinstance(genres_list, str):
                genres = [genre.strip() for genre in genres_list.split(',')]
                for genre in genres:
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        top_n = 15
        top_genres = pd.Series(genre_counts).sort_values(ascending=False).head(top_n)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_genres.index, y=top_genres.values, palette='viridis')
        plt.title('Top Movie Genres by Frequency')
        plt.xlabel('Genre')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_path, "top_genres.png"), bbox_inches='tight')
        plt.close()

    def plot_popularity_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.merged_df['popularity'], bins=50, kde=False)
        plt.title('Distribution of Movie Popularity')
        plt.xlabel('Popularity')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.img_path, "popularity_distribution.png"), bbox_inches='tight')
        plt.close()

        filtered_popularity_df = self.merged_df[self.merged_df['popularity'] < 100].copy()
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_popularity_df['popularity'], bins=50, kde=False)
        plt.title('Distribution of Movie Popularity (Popularity < 100)')
        plt.xlabel('Popularity')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.img_path, "popularity_distribution_lt100.png"), bbox_inches='tight')
        plt.close()

        filtered_popularity_df_low = self.merged_df[self.merged_df['popularity'] < 10].copy()
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_popularity_df_low['popularity'], bins=50, kde=False)
        plt.title('Distribution of Movie Popularity (Popularity < 10)')
        plt.xlabel('Popularity')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.img_path, "popularity_distribution_lt10.png"), bbox_inches='tight')
        plt.close()

    def plot_runtime_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.merged_df['runtime'].dropna(), bins=50, kde=False)
        plt.title('Distribution of Movie Runtimes')
        plt.xlabel('Runtime (minutes)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.img_path, "runtime_distribution.png"), bbox_inches='tight')
        plt.close()

    def plot_production_company_counts(self):
        company_counts = {}
        for companies_list in self.merged_df['production_companies'].dropna():
            if isinstance(companies_list, str):
                companies = [company.strip() for company in companies_list.split(',')]
                for company in companies:
                    if company and company != 'Unknown':
                        company_counts[company] = company_counts.get(company, 0) + 1
        top_n_companies = 15
        top_companies = pd.Series(company_counts).sort_values(ascending=False).head(top_n_companies)
        plt.figure(figsize=(14, 8))
        sns.barplot(x=top_companies.index, y=top_companies.values, palette='viridis')
        plt.title(f'Top {top_n_companies} Production Companies')
        plt.xlabel('Production Company')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_path, "top_production_companies.png"), bbox_inches='tight')
        plt.close()

    def plot_production_country_counts(self):
        country_counts = {}
        for countries_list in self.merged_df['production_countries'].dropna():
            if isinstance(countries_list, str):
                countries = [country.strip() for country in countries_list.split(',')]
                for country in countries:
                    if country and country != 'Unknown':
                        country_counts[country] = country_counts.get(country, 0) + 1
        top_n_countries = 15
        top_countries = pd.Series(country_counts).sort_values(ascending=False).head(top_n_countries)
        plt.figure(figsize=(14, 8))
        sns.barplot(x=top_countries.index, y=top_countries.values, palette='magma')
        plt.title(f'Top {top_n_countries} Production Countries')
        plt.xlabel('Production Country')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_path, "top_production_countries.png"), bbox_inches='tight')
        plt.close()

    def plot_language_counts(self):
        language_counts = {}
        for languages_list in self.merged_df['spoken_languages'].dropna():
            if isinstance(languages_list, str):
                languages = [lang.strip() for lang in languages_list.split(',')]
                for lang in languages:
                    if lang and lang != 'Unknown':
                        language_counts[lang] = language_counts.get(lang, 0) + 1
        language_counts_series = pd.Series(language_counts).sort_values(ascending=False)
        top_languages = language_counts_series.head(15)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_languages.index, y=top_languages.values, palette='viridis')
        plt.title('Top 15 Spoken Languages')
        plt.xlabel('Language')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_path, "top_languages.png"), bbox_inches='tight')
        plt.close()

    def plot_vote_count_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.merged_df['vote_count'], bins=50, kde=False)
        plt.title('Distribution of Movie Vote Counts')
        plt.xlabel('Vote Count')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.img_path, "vote_count_distribution.png"), bbox_inches='tight')
        plt.close()

    def plot_vote_average_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.merged_df['vote_average'], bins=20, kde=False)
        plt.title('Distribution of Movie Vote Averages')
        plt.xlabel('Vote Average')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.img_path, "vote_average_distribution.png"), bbox_inches='tight')
        plt.close()

    def plot_vote_count_vs_average(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.merged_df, x='vote_count', y='vote_average')
        plt.title('Relationship between Vote Count and Vote Average')
        plt.xlabel('Vote Count')
        plt.ylabel('Vote Average')
        plt.savefig(os.path.join(self.img_path, "vote_count_vs_average.png"), bbox_inches='tight')
        plt.close()

    def plot_wordclouds(self):
        copy = self.df.copy()
        copy['title'] = copy['title'].astype('str')
        copy['overview'] = copy['overview'].astype('str')
        title_corpus = ' '.join(copy['title'])
        overview_corpus = ' '.join(copy['overview'])

        title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(title_corpus)
        plt.figure(figsize=(16,8))
        plt.imshow(title_wordcloud)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_path, "wordcloud_title.png"), bbox_inches='tight')
        plt.close()

        overview_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(overview_corpus)
        plt.figure(figsize=(16,8))
        plt.imshow(overview_wordcloud)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_path, "wordcloud_overview.png"), bbox_inches='tight')
        plt.close()

    def plot_world_production_map(self):

        copy = self.df.copy()
        country_counts = copy['production_countries'].value_counts().reset_index()
        country_counts.columns = ['country', 'num_movies']
        country_counts = country_counts[country_counts['country'] != "United States of America"]

        def get_iso3(country_name):
            try:
                return pycountry.countries.lookup(country_name).alpha_3
            except:
                return None

        country_counts['iso_alpha'] = country_counts['country'].apply(get_iso3)
        country_counts = country_counts.dropna(subset=['iso_alpha'])

        data = [go.Choropleth(
            locations = country_counts['iso_alpha'],
            z = country_counts['num_movies'],
            text = country_counts['country'],
            colorscale = [[0,'rgb(255,255,255)'], [1,'rgb(255,0,0)']],
            autocolorscale = False,
            reversescale = False,
            marker = dict(line = dict(color='rgb(180,180,180)', width=0.5)),
            colorbar = dict(title='Production Countries')
        )]

        layout = dict(
            title = 'Production Countries for the MovieLens Movies (Apart from US)',
            geo = dict(
                showframe = False,
                showcoastlines = False,
                projection = dict(type = 'mercator')
            )
        )

        fig = go.Figure(data=data, layout=layout)
<<<<<<< HEAD
        # Save as static image (requires kaleido)
        try:
            # Use plotly.io.write_image for better compatibility
            pio.write_image(fig, os.path.join(self.img_path, "world_production_map.png"))
        except Exception:
            # As a fallback, save as HTML if static image export fails
=======
        try:
            pio.write_image(fig, os.path.join(self.img_path, "world_production_map.png"))
        except Exception:
>>>>>>> 848b2a5 (almost finilaize)
            try:
                fig.write_html(os.path.join(self.img_path, "world_production_map.html"))
            except Exception:
                pass

    def plot_decade_pie(self):
        import plotly.express as px
        copy = self.df.copy()
        copy['release_date'] = pd.to_datetime(copy['release_date'], errors='coerce')
        copy['decade'] = (copy['release_date'].dt.year // 10) * 10
        decade_counts = copy['decade'].value_counts().sort_index().reset_index()
        decade_counts.columns = ['decade', 'num_movies']
        decade_counts['decade'] = decade_counts['decade'].astype(int).astype(str) + "s"
        fig = px.pie(
            decade_counts,
            names='decade',
            values='num_movies',
            title="Movies Distribution by Decade (Release Date)",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
<<<<<<< HEAD
        # Save as static image (requires kaleido)
        try:
            # Use plotly.io.write_image for better compatibility
            pio.write_image(fig, os.path.join(self.img_path, "movies_by_decade_pie.png"))
        except Exception:
            # As a fallback, save as HTML if static image export fails
=======
        try:
            pio.write_image(fig, os.path.join(self.img_path, "movies_by_decade_pie.png"))
        except Exception:
>>>>>>> 848b2a5 (almost finilaize)
            try:
                fig.write_html(os.path.join(self.img_path, "movies_by_decade_pie.html"))
            except Exception:
                pass



    def run_all(self):
        self.plot_rating_distribution()
        self.plot_release_year_distribution()
        self.plot_budget_vs_revenue()
        self.plot_genre_counts()
        self.plot_popularity_distribution()
        self.plot_runtime_distribution()
        self.plot_production_company_counts()
        self.plot_production_country_counts()
        self.plot_language_counts()
        self.plot_vote_count_distribution()
        self.plot_vote_average_distribution()
        self.plot_vote_count_vs_average()
        self.plot_wordclouds()
        self.plot_world_production_map()
        self.plot_decade_pie()
