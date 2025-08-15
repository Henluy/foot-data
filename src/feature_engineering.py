import pandas as pd

def calculate_form(df, window_size=5):
    """
    Calculates the form of each team over a rolling window of matches.
    This version includes fixes for index alignment and merge logic.
    """
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date')

    # Create a unique match ID to join on later
    df = df.reset_index().rename(columns={'index': 'match_id'})

    # Unpivot the dataframe to have one row per team per match
    home_games = df[['match_id', 'Date', 'HomeTeam', 'FTR']].rename(columns={'HomeTeam': 'Team'})
    home_games['points'] = home_games['FTR'].apply(lambda x: 3 if x == 'H' else 1 if x == 'D' else 0)

    away_games = df[['match_id', 'Date', 'AwayTeam', 'FTR']].rename(columns={'AwayTeam': 'Team'})
    away_games['points'] = away_games['FTR'].apply(lambda x: 3 if x == 'A' else 1 if x == 'D' else 0)

    # Combine, sort by team and date to ensure rolling calculations are correct
    all_games = pd.concat([home_games, away_games]).sort_values(['Team', 'Date'])

    # Calculate rolling form for each team
    # The result of groupby().rolling() has a MultiIndex. We need to handle it properly.
    form_series = all_games.groupby('Team')['points'].rolling(window=window_size, min_periods=1).sum().shift(1)

    # Drop the 'Team' level of the index so it can be merged back
    form_series = form_series.reset_index(level=0, drop=True)

    # Assign the form back to the `all_games` dataframe
    all_games['form'] = form_series
    all_games.fillna({'form': 0}, inplace=True)

    # Merge form back into the original dataframe
    # Merge Home Team Form
    home_form_map = all_games.rename(columns={'Team': 'HomeTeam', 'form': 'HomeTeam_Form'})
    merged_df = pd.merge(df, home_form_map[['match_id', 'HomeTeam', 'HomeTeam_Form']], on=['match_id', 'HomeTeam'], how='left')

    # Merge Away Team Form
    away_form_map = all_games.rename(columns={'Team': 'AwayTeam', 'form': 'AwayTeam_Form'})
    merged_df = pd.merge(merged_df, away_form_map[['match_id', 'AwayTeam', 'AwayTeam_Form']], on=['match_id', 'AwayTeam'], how='left')

    # Fill any NaNs that might result from the merges (e.g., first games of the season)
    merged_df.fillna({'HomeTeam_Form': 0, 'AwayTeam_Form': 0}, inplace=True)

    # Drop the temporary match_id
    merged_df.drop(columns=['match_id'], inplace=True)

    return merged_df

def main():
    """
    Loads raw data, engineers features, and saves the processed data.
    """
    print("Starting feature engineering...")
    raw_data_path = 'data/premier_league_2324.csv'
    processed_data_path = 'data/processed_data.csv'

    try:
        df = pd.read_csv(raw_data_path, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}")
        return

    # Calculate form features with the corrected logic
    df_featured = calculate_form(df, window_size=5)

    # Select relevant columns for the model
    final_cols = [
        'Date', 'HomeTeam', 'AwayTeam',
        'FTHG', 'FTAG', 'FTR',
        'HomeTeam_Form', 'AwayTeam_Form'
    ]
    df_final = df_featured[final_cols]

    # Drop duplicates that may have been created during merges
    df_final = df_final.drop_duplicates()

    # Save the processed data
    df_final.to_csv(processed_data_path, index=False)
    print(f"Feature engineering complete. Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    main()
