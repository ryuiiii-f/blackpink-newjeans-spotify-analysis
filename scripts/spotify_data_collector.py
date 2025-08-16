import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import json
from datetime import datetime
import os

print("🎵 MULTI-ARTIST SPOTIFY TOP TRACKS COLLECTOR")
print("="*60)

# 直接填入你的Spotify credentials
SPOTIFY_CLIENT_ID = "17dcf2d694ff47f7aa958e7b10289462"
SPOTIFY_CLIENT_SECRET = "a84d21864bc64143b1831ae44b5fb6b0"

def init_spotify():
    """初始化Spotify API客户端"""
    try:
        client_credentials_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        )
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        print("✅ Spotify API connected successfully")
        return sp
    except Exception as e:
        print(f"❌ Spotify API connection failed: {e}")
        return None

def get_artist_top_tracks(sp, artist_name, limit=10):
    """获取艺人最热门的歌曲"""
    try:
        # 搜索艺人
        print(f"\n🔍 Searching for artist: {artist_name}")
        results = sp.search(q=f"artist:{artist_name}", type='artist', limit=1)
        
        if not results['artists']['items']:
            print(f"❌ Artist {artist_name} not found")
            return None, None
            
        artist = results['artists']['items'][0]
        artist_id = artist['id']
        
        print(f"✅ Found artist: {artist['name']} (ID: {artist_id})")
        print(f"   👥 Followers: {artist['followers']['total']:,}")
        print(f"   📊 Popularity: {artist['popularity']}/100")
        
        # 获取Top Tracks
        print(f"🎵 Getting top {limit} tracks...")
        top_tracks = sp.artist_top_tracks(artist_id, country='US')
        
        if len(top_tracks['tracks']) == 0:
            print(f"❌ No tracks found for {artist_name}")
            return None, None
            
        print(f"✅ Found {len(top_tracks['tracks'])} tracks")
        
        # 显示找到的歌曲列表
        print(f"📋 Top tracks for {artist_name}:")
        for i, track in enumerate(top_tracks['tracks'][:limit], 1):
            print(f"   {i:2d}. {track['name']} (Popularity: {track['popularity']})")
        
        return top_tracks['tracks'][:limit], artist
        
    except Exception as e:
        print(f"❌ Error getting top tracks for {artist_name}: {e}")
        return None, None

def get_track_detailed_info(sp, track):
    """获取单首歌曲的详细信息"""
    try:
        track_info = {
            # 基础信息
            'track_id': track['id'],
            'track_name': track['name'],
            'artist_name': track['artists'][0]['name'],
            'album_name': track['album']['name'],
            'album_id': track['album']['id'],
            'release_date': track['album']['release_date'],
            'duration_ms': track['duration_ms'],
            'duration_minutes': round(track['duration_ms'] / 60000, 2),
            'track_popularity': track['popularity'],
            'explicit': track['explicit'],
            'preview_url': track['preview_url'],
            'spotify_track_url': track['external_urls']['spotify'],
            'track_number': track['track_number'],
            'album_total_tracks': track['album']['total_tracks'],
            'album_type': track['album']['album_type']
        }
        
        # 获取专辑详细信息
        try:
            album = sp.album(track['album']['id'])
            track_info.update({
                'album_popularity': album.get('popularity', 0),
                'album_release_date': album['release_date'],
                'album_external_urls': album['external_urls']['spotify']
            })
        except:
            print(f"   ⚠️ Could not get album details for {track['name']}")
        
        # 获取音频特征
        try:
            features = sp.audio_features([track['id']])[0]
            if features:
                track_info.update({
                    'danceability': features['danceability'],
                    'energy': features['energy'],
                    'valence': features['valence'],
                    'tempo': features['tempo'],
                    'loudness': features['loudness'],
                    'speechiness': features['speechiness'],
                    'acousticness': features['acousticness'],
                    'instrumentalness': features['instrumentalness'],
                    'liveness': features['liveness'],
                    'key': features['key'],
                    'mode': features['mode'],
                    'time_signature': features['time_signature'],
                    'mood_score': (features['valence'] + features['energy']) / 2,
                    'danceability_energy_combo': features['danceability'] * features['energy'],
                    'acoustic_vs_electronic': 1 - features['acousticness']
                })
                print(f"   🎵 Audio features collected")
            else:
                print(f"   ⚠️ Audio features not available")
        except Exception as e:
            print(f"   ⚠️ Audio features error: {e}")
        
        # 获取音频分析摘要
        try:
            analysis = sp.audio_analysis(track['id'])
            track_analysis = analysis.get('track', {})
            sections = analysis.get('sections', [])
            
            track_info.update({
                'num_sections': len(sections),
                'avg_section_duration': sum(s['duration'] for s in sections) / len(sections) if sections else 0,
                'key_confidence': track_analysis.get('key_confidence', 0),
                'mode_confidence': track_analysis.get('mode_confidence', 0),
                'tempo_confidence': track_analysis.get('tempo_confidence', 0),
                'end_of_fade_in': track_analysis.get('end_of_fade_in', 0),
                'start_of_fade_out': track_analysis.get('start_of_fade_out', 0),
                'structural_complexity': len(sections) / (track_analysis.get('duration', 1) / 60) if sections else 0
            })
            print(f"   🔬 Audio analysis collected")
        except Exception as e:
            print(f"   ⚠️ Audio analysis error: {e}")
        
        return track_info
        
    except Exception as e:
        print(f"❌ Error getting track details for {track.get('name', 'Unknown')}: {e}")
        return None

def collect_artist_data(sp, artist_name, limit=10):
    """收集单个艺人的完整数据"""
    print(f"\n{'='*20} {artist_name.upper()} {'='*20}")
    
    # 获取Top Tracks
    top_tracks, artist_info = get_artist_top_tracks(sp, artist_name, limit)
    
    if not top_tracks or not artist_info:
        print(f"❌ Failed to get data for {artist_name}")
        return None
    
    # 收集每首歌的详细信息
    collected_data = []
    
    for i, track in enumerate(top_tracks, 1):
        print(f"\n--- Processing Track {i}/{len(top_tracks)}: {track['name']} ---")
        
        track_data = get_track_detailed_info(sp, track)
        
        if track_data:
            # 添加艺人信息
            track_data.update({
                'artist_id': artist_info['id'],
                'artist_popularity': artist_info['popularity'],
                'artist_followers': artist_info['followers']['total'],
                'artist_genres': ', '.join(artist_info['genres']),
                'artist_external_urls': artist_info['external_urls']['spotify'],
                'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'track_rank': i  # Top tracks排名
            })
            
            # 计算性能分数
            track_data['performance_score'] = (
                track_data['track_popularity'] * 1.2 + 
                track_data['artist_popularity'] * 0.8
            )
            track_data['relative_popularity'] = track_data['track_popularity'] / track_data['artist_popularity']
            
            collected_data.append(track_data)
            
            print(f"   ✅ {track['name']} - Popularity: {track['popularity']}/100")
            if 'danceability' in track_data:
                print(f"      🎵 Danceability: {track_data['danceability']:.3f}, Energy: {track_data['energy']:.3f}")
            print(f"      📊 Performance Score: {track_data['performance_score']:.1f}")
        else:
            print(f"   ❌ Failed to collect data for {track['name']}")
    
    return collected_data

def save_artist_data(artist_data, artist_name):
    """保存艺人数据到文件"""
    if not artist_data:
        print(f"❌ No data to save for {artist_name}")
        return
    
    # 创建目录
    os.makedirs("../../data/processed", exist_ok=True)
    
    # 文件名格式化
    safe_name = artist_name.lower().replace(" ", "_")
    
    # 转换为DataFrame
    df = pd.DataFrame(artist_data)
    
    # 保存详细数据
    detailed_file = f"../../data/processed/{safe_name}_spotify_top_tracks.csv"
    df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
    
    # 创建摘要数据
    summary_columns = [
        'track_name', 'track_rank', 'track_popularity', 'artist_popularity',
        'duration_minutes', 'album_name', 'album_type', 'performance_score',
        'relative_popularity', 'release_date'
    ]
    
    # 如果有音频特征，添加到摘要中
    if 'danceability' in df.columns:
        summary_columns.extend(['danceability', 'energy', 'valence', 'tempo', 'mood_score'])
    
    summary_df = df[summary_columns].copy()
    summary_file = f"../../data/processed/{safe_name}_performance_summary.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    
    # 如果有音频特征，单独保存
    if 'danceability' in df.columns:
        audio_features_columns = [
            'track_name', 'track_rank', 'danceability', 'energy', 'valence', 'tempo',
            'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'key', 'mode', 'time_signature', 'mood_score', 'danceability_energy_combo',
            'acoustic_vs_electronic', 'structural_complexity'
        ]
        
        audio_df = df[audio_features_columns].copy()
        audio_file = f"../../data/processed/{safe_name}_audio_features.csv"
        audio_df.to_csv(audio_file, index=False, encoding='utf-8-sig')
        
        print(f"   🎵 Audio features saved: {audio_file}")
    
    print(f"   📊 Detailed data saved: {detailed_file}")
    print(f"   📈 Summary saved: {summary_file}")
    print(f"   ✅ Total tracks: {len(artist_data)}")
    
    # 显示数据统计
    print(f"\n📊 {artist_name.upper()} DATA SUMMARY:")
    print(f"   👑 Artist Popularity: {artist_data[0]['artist_popularity']}/100")
    print(f"   👥 Followers: {artist_data[0]['artist_followers']:,}")
    print(f"   🎵 Top Track: {artist_data[0]['track_name']} (Rank #1)")
    print(f"   📈 Avg Track Popularity: {df['track_popularity'].mean():.1f}")
    print(f"   ⏱️ Avg Duration: {df['duration_minutes'].mean():.2f} minutes")
    if 'mood_score' in df.columns:
        print(f"   😊 Avg Mood Score: {df['mood_score'].mean():.3f}")

def main():
    """主函数"""
    sp = init_spotify()
    if not sp:
        print("💡 Please check your Spotify API credentials")
        return
    
    # 要收集的艺人列表
    artists = [
        {"name": "BLACKPINK", "limit": 10},
        {"name": "NewJeans", "limit": 10}
    ]
    
    print(f"\n🎯 COLLECTING TOP TRACKS DATA")
    print(f"   📋 Artists: {len(artists)}")
    print(f"   🎵 Tracks per artist: 10")
    print(f"   📊 Total tracks: {sum(artist['limit'] for artist in artists)}")
    
    # 收集每个艺人的数据
    for artist in artists:
        artist_data = collect_artist_data(sp, artist["name"], artist["limit"])
        
        if artist_data:
            save_artist_data(artist_data, artist["name"])
        else:
            print(f"❌ Failed to collect data for {artist['name']}")
    
    print(f"\n🎉 DATA COLLECTION COMPLETE!")
    print(f"📁 Files saved in: ../../data/processed/")
    print(f"   🖤 BLACKPINK: blackpink_spotify_top_tracks.csv")
    print(f"   💗 NewJeans: newjeans_spotify_top_tracks.csv")
    
    print(f"\n🔗 NEXT STEPS:")
    print(f"   1. 📊 Load data for analysis")
    print(f"   2. 🎯 Compare top tracks characteristics") 
    print(f"   3. 📈 Analyze global influence patterns")
    print(f"   4. 🎨 Create comparison visualizations")

if __name__ == "__main__":
    main()
