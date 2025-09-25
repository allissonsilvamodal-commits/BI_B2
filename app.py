import os
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as se
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
import openai

st.set_page_config(layout="wide")  # Layout amplo

# --- OpenAI ---
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if openai_api_key:
    openai.api_key = openai_api_key
    client = openai.OpenAI(api_key=openai.api_key)
    st.sidebar.success("OpenAI client initialized.")
else:
    st.sidebar.warning("OpenAI API key not found. OpenAI features will be disabled.")
    client = None

# --- Google Sheets ---
try:
    google_creds = {
        key: (value.replace("\\n", "\n") if key == "private_key" else value)
        for key, value in st.secrets["google"].items()
    }
    creds = Credentials.from_service_account_info(google_creds)
    gc = gspread.authorize(creds)
    st.sidebar.success("Google Sheets authenticated via secrets.")
except Exception as e:
    st.sidebar.error(f"Erro na autenticação do Google Sheets: {e}")
    gc = None

@st.cache_data # Cache the data loading for performance (removed allow_output_mutation=True)
def load_data(_gspread_client): # Added underscore to prevent hashing
    """Loads data from Google Sheet using the provided gspread client."""
    if _gspread_client is None:
        st.warning("Cliente gspread não disponível. Não foi possível carregar os dados.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        B2sheet = _gspread_client.open('BI_B2')
        page = B2sheet.sheet1
        all_data = page.get_all_values()
        df = pd.DataFrame(all_data[1:], columns=all_data[0])

        # Limpeza e pré-processing
        value_cols = ['Valor', 'Valor Rec.', 'Valor fechamento', 'Valor rec. fechamento']
        for col in value_cols:
            df[col] = df[col].astype(str).str.replace('R$', '', regex=False).str.replace(',', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

        date_cols = ['Data de abertura', 'Data fechamento']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce', format='%d/%m/%Y %H:%M:%S')

        # Extract unique opportunity identifier (OC + number) from 'Título'
        def extract_oc_identifier(title):
            if isinstance(title, str):
                match = re.search(r'(OC\s*\d+)', title, re.IGNORECASE)
                if match:
                    return match.group(1).replace(" ", "") # Remove space in "OC "
                else:
                    # Also check for "CTE" as it appears in titles and might be identifiers
                    match_cte = re.search(r'(CTE\s*\d+)', title, re.IGNORECASE)
                    if match_cte:
                         return match_cte.group(1).replace(" ", "")
            return None # Return None if no pattern is found

        df['OC_Identifier'] = df['Título'].apply(extract_oc_identifier)


        # Transformation and Feature Engineering for BI
        # Ensure date columns are not NaT before accessing dt properties
        df['Mes de Abertura'] = df['Data de abertura'].dt.month.fillna(0).astype(int) if pd.api.types.is_datetime64_any_dtype(df['Data de abertura']) else 0
        df['Ano de Abertura'] = df['Data de abertura'].dt.year.fillna(0).astype(int) if pd.api.types.is_datetime64_any_dtype(df['Data de abertura']) else 0
        df['Mes de Fechamento'] = df['Data fechamento'].dt.month.fillna(0).astype(int) if pd.api.types.is_datetime64_any_dtype(df['Data fechamento']) else 0
        df['Ano de Fechamento'] = df['Data fechamento'].dt.year.fillna(0).astype(int) if pd.api.types.is_datetime64_any_dtype(df['Data fechamento']) else 0

        # Create a 'MonthYear' column for time series analysis
        df['MonthYear_Abertura'] = df['Data de abertura'].dt.to_period('M') if pd.api.types.is_datetime64_any_dtype(df['Data de abertura']) else None
        df['MonthYear_Fechamento'] = df['Data fechamento'].dt.to_period('M') if pd.api.types.is_datetime64_any_dtype(df['Data fechamento']) else None

        # Extract Hour of Day from 'Data de abertura'
        # Ensure 'Data de abertura' is not NaT before extracting hour
        # Only extract hour if Data de abertura is a valid datetime
        df['Hour_of_Day_Abertura'] = df['Data de abertura'].apply(lambda x: x.hour if pd.notna(x) else -1).astype(int)


        # Calculate Time in Stage for timeline analysis using OC_Identifier
        df_timeline = df[['OC_Identifier', 'Estágio', 'Data de abertura', 'Data fechamento']].copy()
        # Drop rows where OC_Identifier could not be extracted or Data de abertura is missing
        df_timeline.dropna(subset=['OC_Identifier', 'Data de abertura'], inplace=True)
        df_timeline = df_timeline.sort_values(by=['OC_Identifier', 'Data de abertura'])

        current_time = pd.to_datetime('now') # Timezone-naive for consistency
        df_timeline['Time_in_Stage'] = (df_timeline['Data fechamento'] - df_timeline['Data de abertura']).dt.total_seconds() / 3600 # Time in hours

        df_timeline['Time_in_Stage'] = df_timeline.apply(
            lambda row: (current_time - row['Data de abertura']).total_seconds() / 3600 if pd.isna(row['Data fechamento']) else row['Time_in_Stage'],
            axis=1
        )

        # Format 'Time_in_Stage' for display to include minutes
        def format_time_in_stage(hours):
            if pd.isna(hours):
                return "N/A"
            total_minutes = int(hours * 60)
            days = total_minutes // (24 * 60)
            remaining_minutes_after_days = total_minutes % (24 * 60)
            hours = remaining_minutes_after_days // 60
            minutes = remaining_minutes_after_days % 60
            return f"{days} days, {hours} hours, {minutes} minutes"

        df_timeline['Time_in_Stage_Formatted'] = df_timeline['Time_in_Stage'].apply(format_time_in_stage)


        return df, df_timeline

    except Exception as e:
        st.error(f"Erro ao carregar dados do Google Sheet: {e}")
        return pd.DataFrame(), pd.DataFrame() # Return empty dataframes on error


# Load data (pass the gc client)
df, df_timeline = load_data(gc)

# --- Placeholder Interaction Data ---
# Create a simple placeholder DataFrame for site interaction data
interaction_data = {
    'User': ['User A', 'User B', 'User C', 'User A', 'User B', 'User D', 'User A'],
    'Interactions': [10, 15, 8, 12, 18, 5, 11]
}
df_interaction = pd.DataFrame(interaction_data)

# Aggregate interaction data by user
df_agg_interaction = df_interaction.groupby('User')['Interactions'].sum().reset_index()


# --- Authentication Logic ---
# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Placeholder credentials (replace with secure storage in production)
VALID_USERNAME = st.secrets["login"]["username"]
VALID_PASSWORD = st.secrets["login"]["password"]

def authenticate(username, password):
    """Simple function to check placeholder credentials."""
    return username == VALID_USERNAME and password == VALID_PASSWORD

# --- Multi-page Navigation ---
st.sidebar.title("Navegação")

# Determine the current page based on authentication status and sidebar selection
if st.session_state['authenticated']:
    # If authenticated, show all pages in the sidebar radio
    # Initialize the page in session state if not set, default to "Página Inicial"
    if 'page' not in st.session_state:
        st.session_state['page'] = "Página Inicial"

    page = st.sidebar.radio("Ir para:", ["Página Inicial", "Painel Geral", "Relatório de Oportunidade"], index=["Página Inicial", "Painel Geral", "Relatório de Oportunidade"].index(st.session_state['page']))

    # Update session state page when sidebar changes
    st.session_state['page'] = page

    # Add Logout button if authenticated
    if st.sidebar.button("Logout"):
        st.session_state['authenticated'] = False
        st.session_state['page'] = "Login" # Redirect to login on logout
        st.rerun() # Rerun to show login page
else:
    # If not authenticated, force the page to "Login" and hide other options
    page = "Login"
    st.session_state['page'] = "Login" # Ensure session state reflects the login page


# --- Refresh Button (only show if authenticated) ---
if st.session_state['authenticated']:
    if st.sidebar.button("Atualizar Dados"):
        st.cache_data.clear() # Clear the cache for load_data
        st.rerun() # Rerun the app to load fresh data


# Main app logic within a try-except block for general errors
try:
    if page == "Login":
        # --- Login Page ---
        st.title("Login")
        st.markdown("Por favor, insira suas credenciais para acessar o painel.")

        # Create input fields for username and password
        with st.form("login_form"):
            username = st.text_input("Nome de Usuário")
            password = st.text_input("Senha", type="password")
            login_button = st.form_submit_button("Login")

            if login_button:
                if authenticate(username, password):
                    st.session_state['authenticated'] = True
                    st.success("Login bem-sucedido!")
                    # Redirect to "Página Inicial" after successful login
                    st.session_state['page'] = "Página Inicial" # Set the page in session state
                    st.rerun() # Rerun to show the initial page content
                else:
                    st.error("Nome de usuário ou senha inválidos.")

    # Conditionally render other pages based on authentication
    elif st.session_state['authenticated']:
        if page == "Página Inicial":
            # --- Página Inicial (after login) ---
            st.title("Bem-vindo ao Painel de BI Operacional")

            st.markdown("""
                Este painel interativo oferece insights valiosos sobre suas oportunidades de negócios.
                Navegue pelas seções usando o menu ao lado para explorar:

                *   **Painel Geral**: Uma visão abrangente das métricas e distribuições de oportunidades.
                *   **Relatório de Oportunidade Individual**: Análise detalhada da linha do tempo e informações de oportunidades específicas.

                Utilize os filtros em cada página para personalizar sua análise.
            """)

            # Add some visual elements or structure
            st.subheader("Visão Geral Rápida")
            col_intro1, col_intro2 = st.columns(2)

            with col_intro1:
                st.info("Clique no 'Painel Geral' para começar a explorar os dados agregados.")

            with col_intro2:
                st.info("Clique em 'Relatório de Oportunidade' para analisar oportunidades específicas.")

            st.markdown("---") # Separator

            st.subheader("Interação do Site por Usuário (Dados de Exemplo)")
            # Create and display the interaction graph using the placeholder data
            if not df_agg_interaction.empty:
                fig_interaction = px.bar(df_agg_interaction, x='User', y='Interactions',
                                         title='Total de Interações do Site por Usuário (Dados de Exemplo)',
                                         template='plotly_white',
                                         color='User', # Color by user
                                         color_discrete_sequence=px.colors.qualitative.Plotly) # Use a standard Plotly color scale

                fig_interaction.update_layout(xaxis_title="Usuário", yaxis_title="Número Total de Interações")
                st.plotly_chart(fig_interaction, use_container_width=True)
            else:
                st.info("Nenhum dado de interação de usuário disponível para exibir.")

            st.markdown("---") # Separator

            st.subheader("Sobre o Projeto")
            st.markdown("""
                Este projeto de Business Intelligence foi desenvolvido para fornecer uma visão clara e acionável
                sobre o desempenho das oportunidades de negócios, identificar gargalos no processo e facilitar a tomada de decisão.

                **Idealizador e Supervisor de Projetos:**
                Allisson Silva

                **Contato:**
                *   Telefone: +55 81 9760-0051
                *   Gmail: allisson.silva.modal@gmail.com
                *   Outlook: Allisson.silva@logmultimodal.com.br

                Sinta-se à vontade para entrar em contato para feedback, sugestões ou colaboração.
            """)


        elif page == "Painel Geral":
            # --- Painel Geral ---
            st.title("Painel de BI Operacional - Geral")

            st.markdown("""
            Este painel apresenta insights gerais sobre os dados operacionais,
            permitindo acompanhar o desempenho e a distribuição dos negócios.
            """)

            if not df.empty:
                # Add filters to the sidebar
                st.sidebar.subheader("Filtros do Painel Geral")

                # Date range filter for 'Data de abertura'
                if not df['Data de abertura'].empty:
                    min_date_abertura = df['Data de abertura'].min().date()
                    max_date_abertura = df['Data de abertura'].max().date()

                    start_date = st.sidebar.date_input("Data de Abertura (Início)", min_value=min_date_abertura, max_value=max_date_abertura, value=min_date_abertura)
                    end_date = st.sidebar.date_input("Data de Abertura (Fim)", min_value=min_date_abertura, max_value=max_date_abertura, value=max_date_abertura)

                    # Convert selected dates to datetime objects for filtering
                    start_datetime = pd.to_datetime(start_date)
                    end_datetime = pd.to_datetime(end_date)

                    # Apply date filter
                    filtered_df = df[(df['Data de abertura'] >= start_datetime) & (df['Data de abertura'] <= end_datetime)].copy()
                else:
                    filtered_df = df.copy() # If date column is empty, use the original df


                # Filter by Estado
                selected_estados = st.sidebar.multiselect("Selecionar Estado:", filtered_df['Estado'].unique(), filtered_df['Estado'].unique())
                filtered_df = filtered_df[filtered_df['Estado'].isin(selected_estados)]

                # Filter by Responsável (Initial filter)
                selected_responsaveis_sidebar = st.sidebar.multiselect("Selecionar Responsável (Filtro Inicial):", filtered_df['Responsável'].unique(), filtered_df['Responsável'].unique())
                filtered_df = filtered_df[filtered_df['Responsável'].isin(selected_responsaveis_sidebar)]

                # Filter by Estágio
                selected_estagios = st.sidebar.multiselect("Selecionar Estágio:", filtered_df['Estágio'].unique(), filtered_df['Estágio'].unique())
                filtered_df = filtered_df[filtered_df['Estágio'].isin(selected_estagios)]

                # Add filter by OC_Identifier for the general dashboard
                opportunity_identifiers = filtered_df['OC_Identifier'].dropna().unique()
                selected_opportunity_identifier_general = st.sidebar.selectbox(
                    "Filtrar por Oportunidade (OC + Número ou CTE + Número):",
                    ['Todos'] + list(opportunity_identifiers)
                )

                if selected_opportunity_identifier_general != 'Todos':
                     filtered_df = filtered_df[filtered_df['OC_Identifier'] == selected_opportunity_identifier_general].copy()


                # Agregações para visualizações (using filtered_df)
                # Aggregate to count unique OC_Identifier per Responsável
                df_agg_responsavel_count = filtered_df.groupby('Responsável')['OC_Identifier'].nunique().reset_index()
                df_agg_responsavel_count.rename(columns={'OC_Identifier': 'Unique Opportunity Count'}, inplace=True)


                # Aggregate to count unique OC_Identifier per Estado and MonthYear_Abertura
                df_agg_estado_mes_count = filtered_df.groupby(['Estado', 'MonthYear_Abertura'])['OC_Identifier'].nunique().reset_index()
                df_agg_estado_mes_count.rename(columns={'OC_Identifier': 'Unique Opportunity Count'}, inplace=True)
                df_agg_estado_mes_count['MonthYear_Abertura'] = df_agg_estado_mes_count['MonthYear_Abertura'].astype(str) # Convert to string for plotting


                # Filter for 'Ganha' deals for specific visualizations
                ganha_df_filtered = filtered_df[filtered_df['Estado'] == 'Ganha'].copy()

                # --- Display Key Metrics (KPIs) ---
                st.subheader("Resumo Geral")
                col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

                total_opportunities = filtered_df['OC_Identifier'].nunique() if not filtered_df.empty else 0
                total_won_value = ganha_df_filtered['Valor'].sum() if not ganha_df_filtered.empty else 0
                win_rate = (len(ganha_df_filtered) / total_opportunities * 100) if total_opportunities > 0 else 0


                col_kpi1.metric("Total Oportunidades Únicas", total_opportunities)
                col_kpi2.metric("Valor Total Ganho", f"R$ {total_won_value:,.2f}")
                col_kpi3.metric("Taxa de Sucesso", f"{win_rate:.2f}%")


                # --- Visualizations ---
                st.subheader("Análise de Oportunidades e Valor")

                # Use columns to arrange the plots side-by-side
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Quantidade de Oportunidades Únicas por Responsável")
                    if not df_agg_responsavel_count.empty:
                        # Using Plotly Express for interactivity and improved style
                        fig1 = px.bar(df_agg_responsavel_count, x='Responsável', y='Unique Opportunity Count',
                                      title='Quantidade de Oportunidades Únicas por Responsável',
                                      color='Responsável', # Add color by responsible for visual distinction
                                      template='plotly_white', # Use a clean template
                                      color_discrete_sequence=px.colors.qualitative.Set2) # Use a different color scale

                        fig1.update_layout(xaxis_title="Responsável", yaxis_title="Contagem Única de Oportunidades") # Improved axis titles

                        # Capture selection - Keep for potential filtering later
                        selected_points = st.plotly_chart(fig1, use_container_width=True, on_select="rerun")

                        # Filter based on selection (using the original filtered_df and the selection from the chart)
                        # This filtering logic is already in place in the original code, so we keep it.
                        if selected_points and selected_points.selection and selected_points.selection.points:
                             selected_responsaveis_chart = [point['x'] for point in selected_points.selection.points]
                             # Apply filtering to the original filtered_df based on chart selection
                             filtered_df_chart_selection = filtered_df[filtered_df['Responsável'].isin(selected_responsaveis_chart)].copy()
                        else:
                             filtered_df_chart_selection = filtered_df.copy() # Use sidebar filtered data if no chart selection

                    else:
                        st.info("Nenhum dado disponível para 'Quantidade de Oportunidades Únicas por Responsável' com os filtros selecionados.")
                        filtered_df_chart_selection = filtered_df.copy() # Use sidebar filtered data if no chart data


            with col2:
                 # Recalculate aggregated data based on chart selection
                 df_agg_estado_mes_count_filtered = filtered_df_chart_selection.groupby(['Estado', 'MonthYear_Abertura'])['OC_Identifier'].nunique().reset_index()
                 df_agg_estado_mes_count_filtered.rename(columns={'OC_Identifier': 'Unique Opportunity Count'}, inplace=True)
                 df_agg_estado_mes_count_filtered['MonthYear_Abertura'] = df_agg_estado_mes_count_filtered['MonthYear_Abertura'].astype(str) # Convert to string for plotting

                 st.subheader("Quantidade de Negócios Únicos por Estado e Mês de Abertura")
                 if not df_agg_estado_mes_count_filtered.empty:
                     # Using Plotly Express for the improved chart
                     fig2 = px.bar(df_agg_estado_mes_count_filtered, x='MonthYear_Abertura', y='Unique Opportunity Count', color='Estado',
                                   title='Quantidade de Negócios Únicos por Estado e Mês de Abertura',
                                   barmode='group', # Use 'group' or 'stack'
                                   template='plotly_white', # Use a clean template
                                   color_discrete_sequence=px.colors.qualitative.Pastel) # Use a different color scale

                     fig2.update_layout(xaxis_title="Mês/Ano de Abertura", yaxis_title="Quantidade de Oportunidades Únicas")
                     st.plotly_chart(fig2, use_container_width=True)

                 else:
                      st.info("Nenhum dado disponível para 'Quantidade de Negócios Únicos por Estado e Mês de Abertura' com os filtros selecionados.")

            # Separate row for the heatmap
            st.subheader("Heatmap: Oportunidades por Etapa e Hora de Abertura")
            if not df_timeline.empty:
                # Filter timeline data based on the date range selected for the main dashboard
                df_timeline_filtered_for_heatmap = df_timeline[(df_timeline['Data de abertura'] >= start_datetime) & (df_timeline['Data de abertura'] <= end_datetime)].copy()

                # Further filter timeline data based on chart selection and Estado/Estágio filters
                if selected_points and selected_points.selection and selected_points.selection.points:
                     selected_oc_identifiers_chart = filtered_df_chart_selection['OC_Identifier'].unique()
                     df_timeline_filtered_for_heatmap = df_timeline_filtered_for_heatmap[df_timeline_filtered_for_heatmap['OC_Identifier'].isin(selected_oc_identifiers_chart)].copy()
                df_timeline_filtered_for_heatmap = df_timeline_filtered_for_heatmap[df_timeline_filtered_for_heatmap['Estágio'].isin(selected_estagios)].copy()

                # Filter heatmap data by selected OC_Identifier for the general dashboard
                if selected_opportunity_identifier_general != 'Todos':
                     df_timeline_filtered_for_heatmap = df_timeline_filtered_for_heatmap[df_timeline_filtered_for_heatmap['OC_Identifier'] == selected_opportunity_identifier_general].copy()


                # Aggregate data for the heatmap (count unique OC_Identifier per Estágio and Hour_of_Day_Abertura)
                # Ensure 'Hour_of_Day_Abertura' is extracted for the filtered timeline data
                df_timeline_filtered_for_heatmap['Hour_of_Day_Abertura'] = df_timeline_filtered_for_heatmap['Data de abertura'].dt.hour

                if not df_timeline_filtered_for_heatmap.empty:
                     heatmap_data = df_timeline_filtered_for_heatmap.groupby(['Estágio', 'Hour_of_Day_Abertura'])['OC_Identifier'].nunique().unstack(fill_value=0)

                     if not heatmap_data.empty:
                        # Use go.Heatmap to create the heatmap with improved style and colorscale
                        fig_heatmap = go.Figure(data=go.Heatmap(
                               z=heatmap_data.values,
                               x=heatmap_data.columns,
                               y=heatmap_data.index,
                               colorscale='Portland')) # Use a different color scale

                        fig_heatmap.update_layout(title='Oportunidades por Etapa e Hora de Abertura',
                                                  xaxis_title='Hora do Dia',
                                                  yaxis_title='Etapa',
                                                  template='plotly_white') # Use a clean template

                        st.plotly_chart(fig_heatmap, use_container_width=True)
                     else:
                          st.info("Nenhum dado agregado disponível para o Heatmap com os filtros selecionados.")
                else:
                     st.info("Nenhum dado de timeline disponível para o Heatmap com os filtros selecionados.")
            else:
                 st.info("Dados de timeline não disponíveis para o Heatmap.")


            # Another row for the distribution plots
            st.subheader("Análise de Estágios")
            # Use a single column for the remaining distribution plot after removing one
            col5, = st.columns(1)

            # Removed the "Distribuição do Valor para Negócios Ganhos" chart
            # with col5:
            #      if not ganha_df_filtered.empty:
            #          st.subheader("Distribuição do Valor para Negócios Ganhos (Filtered)")
            #          fig3 = px.histogram(ganha_df_filtered, x='Valor', nbins=50,
            #                              title='Distribuição do Valor para Negócios Ganhos',
            #                              color_discrete_sequence=['indianred'],
            #                              template='plotly_white')
            #          st.plotly_chart(fig3, use_container_width=True)
            #      else:
            #          st.info("Nenhum dado disponível para visualizações de Negócios Ganhos com os filtros selecionados.")

            # Shift the second plot to the single column (col5)
            with col5:
                if not filtered_df_chart_selection.empty:
                    # Distribution of all stages (not just 'Ganha') with improved style and colors
                    st.subheader("Distribuição de Todos os Estágios (Filtered)")
                    # Convert to Plotly bar chart
                    stage_counts = filtered_df_chart_selection['Estágio'].value_counts().reset_index()
                    stage_counts.columns = ['Estágio', 'Count']
                    fig4 = px.bar(stage_counts, x='Estágio', y='Count',
                                  title='Distribuição de Todos os Estágios',
                                  color='Estágio', # Color by stage
                                  template='plotly_white', # Use a clean template
                                  color_discrete_sequence=px.colors.qualitative.Set3) # Use a different color scale
                    fig4.update_layout(xaxis_title="Estágio", yaxis_title="Contagem")
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("Nenhum dado disponível para 'Distribuição de Todos os Estágios' com os filtros selecionados.")


            # Final row for the timeline analysis metrics
            st.subheader("Análise de Tempo Médio por Estágio (Filtered)")
            if not df_timeline.empty:
                # Filter timeline data based on the date range selected for the main dashboard
                df_timeline_filtered = df_timeline[(df_timeline['Data de abertura'] >= start_datetime) & (df_timeline['Data de abertura'] <= end_datetime)].copy()

                # Further filter timeline data based on chart selection and Estado/Estágio filters
                if selected_points and selected_points.selection and selected_points.selection.points:
                     # Filter timeline data based on selected OC_Identifiers from the chart selection
                     selected_oc_identifiers_chart = filtered_df_chart_selection['OC_Identifier'].unique()
                     df_timeline_filtered = df_timeline_filtered[df_timeline_filtered['OC_Identifier'].isin(selected_oc_identifiers_chart)].copy()
                df_timeline_filtered = df_timeline_filtered[df_timeline_filtered['Estágio'].isin(selected_estagios)].copy()

                # Filter timeline data by selected OC_Identifier for the general dashboard
                if selected_opportunity_identifier_general != 'Todos':
                     df_timeline_filtered = df_timeline_filtered[df_timeline_filtered['OC_Identifier'] == selected_opportunity_identifier_general].copy()


                if not df_timeline_filtered.empty:
                    df_agg_time_per_stage_avg = df_timeline_filtered.groupby('Estágio')['Time_in_Stage'].mean().reset_index()
                    df_agg_time_per_stage_avg = df_agg_time_per_stage_avg.sort_values(by='Time_in_Stage', ascending=False)

                    # Format 'Time_in_Stage' for display to include minutes
                    def format_time_in_stage(hours):
                        if pd.isna(hours):
                            return "N/A"
                        total_minutes = int(hours * 60)
                        days = total_minutes // (24 * 60)
                        remaining_minutes_after_days = total_minutes % (24 * 60)
                        hours = remaining_minutes_after_days // 60
                        minutes = remaining_minutes_after_days % 60
                        return f"{days} days, {hours} hours, {minutes} minutes"

                    df_agg_time_per_stage_avg['Average Time in Stage'] = df_agg_time_per_stage_avg['Time_in_Stage'].apply(format_time_in_stage)


                    st.write("Tempo Médio em Cada Estágio:")
                    st.dataframe(df_agg_time_per_stage_avg[['Estágio', 'Average Time in Stage']])

                    # Visualization of average time per stage (using the numerical value) with improved style and colors
                    st.subheader("Tempo Médio por Estágio Visualização (Filtered)")
                    # Convert to Plotly bar chart
                    fig5 = px.bar(df_agg_time_per_stage_avg, x='Estágio', y='Time_in_Stage',
                                  title='Tempo Médio por Estágio (Filtrado)',
                                  color='Estágio', # Color by stage
                                  template='plotly_white', # Use a clean template
                                  color_discrete_sequence=px.colors.qualitative.Vivid) # Use a different color scale
                    fig5.update_layout(xaxis_title='Estágio', yaxis_title='Tempo Médio (horas)')
                    st.plotly_chart(fig5, use_container_width=True)

                else:
                     st.info("Nenhum dado disponível para Análise de Tempo Médio por Estágio com os filtros selecionados.")
            else:
                st.info("Dados de timeline não disponíveis.")


        elif page == "Relatório de Oportunidade":
            # --- Relatório de Oportunidade Individual ---
            st.title("Relatório de Oportunidade Individual")

            st.markdown("""
            Selecione um identificador de Oportunidade (OC + Número ou CTE + Número)
            para visualizar sua linha do tempo e detalhes.
            """)

            # Error handling for empty df or df_timeline
            if df.empty or df_timeline.empty:
                st.warning("Dados de oportunidade ou linha do tempo não disponíveis. Por favor, verifique a conexão com o Google Sheet.")
            else:
                try:
                    # Use OC_Identifier for selection
                    opportunity_identifiers = df['OC_Identifier'].dropna().unique()

                    if len(opportunity_identifiers) == 0:
                        st.info("Nenhum identificador de oportunidade único encontrado nos dados.")
                    else:
                        selected_opportunity_identifier = st.selectbox("Selecionar Oportunidade (OC + Número ou CTE + Número):", opportunity_identifiers)

                        st.subheader(f"Detalhes e Linha do Tempo para: {selected_opportunity_identifier}")

                        try:
                            # Filter main df for the selected opportunity identifier
                            opportunity_details_df = df[df['OC_Identifier'] == selected_opportunity_identifier]

                            if opportunity_details_df.empty:
                                st.warning(f"Nenhum detalhe encontrado para: {selected_opportunity_identifier} no DataFrame principal.")
                            else:
                                opportunity_details = opportunity_details_df.iloc[0]

                                # Use columns for a structured layout
                                col_info1, col_info2 = st.columns(2)

                                with col_info1:
                                    st.write("**ID:**", opportunity_details.get('ID', 'N/A'))
                                    st.write("**Título:**", opportunity_details.get('Título', 'N/A'))
                                    st.write("**Responsável:**", opportunity_details.get('Responsável', 'N/A'))
                                    st.write("**Estado:**", opportunity_details.get('Estado', 'N/A'))
                                    st.write("**Estágio Atual:**", opportunity_details.get('Estágio', 'N/A')) # Display current stage

                                with col_info2:
                                    # Add error handling for potential non-numeric 'Valor' before formatting
                                    valor_display = "N/A"
                                    if pd.notna(opportunity_details.get('Valor')) and pd.api.types.is_numeric_dtype(opportunity_details.get('Valor')):
                                        valor_display = f"R$ {opportunity_details['Valor']:,.2f}"
                                    st.write("**Valor:**", valor_display)

                                    st.write("**Origem:**", opportunity_details.get('Origem', 'N/A'))
                                    st.write("**Prob %:**", opportunity_details.get('Prob %', 'N/A'))
                                    st.write("**OC:**", opportunity_details.get('OC', 'N/A')) # Use .get for safer access


                                st.subheader("Datas Principais")
                                col_dates1, col_dates2 = st.columns(2)
                                with col_dates1:
                                    st.write("**Data de Abertura:**", opportunity_details['Data de abertura'].strftime('%d/%m/%Y %H:%M:%S') if pd.notna(opportunity_details['Data de abertura']) else "N/A")
                                with col_dates2:
                                    st.write("**Data de Fechamento:**", opportunity_details['Data fechamento'].strftime('%d/%m/%Y %H:%M:%S') if pd.notna(opportunity_details['Data fechamento']) else "N/A")

                                # Use an expander for Closing Details (if available)
                                if pd.notna(opportunity_details.get('Data fechamento')):
                                    with st.expander("Detalhes de Fechamento"):
                                        # Add error handling for potential non-numeric closing values
                                        valor_fechamento_display = "N/A"
                                        if pd.notna(opportunity_details.get('Valor fechamento')) and pd.api.types.is_numeric_dtype(opportunity_details.get('Valor fechamento')):
                                            valor_fechamento_display = f"R$ {opportunity_details['Valor fechamento']:,.2f}"
                                        st.write("**Valor Fechamento:**", valor_fechamento_display)

                                        valor_rec_fechamento_display = "N/A"
                                        if pd.notna(opportunity_details.get('Valor rec. fechamento')) and pd.api.types.is_numeric_dtype(opportunity_details.get('Valor rec. fechamento')):
                                            valor_rec_fechamento_display = f"R$ {opportunity_details['Valor rec. fechamento']:,.2f}"
                                        st.write("**Valor Rec. Fechamento:**", valor_rec_fechamento_display)

                                        st.write("**Razão de Fechamento:**", opportunity_details.get('Razão de fechamento', 'N/A')) # Use .get
                                        st.write("**Observação de Fechamento:**", opportunity_details.get('Observação de fechamento', 'N/A')) # Use .get


                                try:
                                    # Filter timeline for the selected opportunity identifier
                                    opportunity_timeline = df_timeline[df_timeline['OC_Identifier'] == selected_opportunity_identifier].copy()

                                    if not opportunity_timeline.empty:
                                        st.subheader("Linha do Tempo da Oportunidade")

                                        # Display timeline using a more visually appealing table format
                                        display_timeline_cols = ['Estágio', 'Data de abertura', 'Data fechamento', 'Time_in_Stage_Formatted']
                                        st.dataframe(opportunity_timeline[display_timeline_cols], key=f"timeline_data_{selected_opportunity_identifier}") # Add unique key
                                    else:
                                        st.info(f"Nenhum dado de linha do tempo encontrado para: {selected_opportunity_identifier}")

                                except Exception as e:
                                    st.error(f"Erro ao processar ou exibir dados de linha do tempo para {selected_opportunity_identifier}: {e}")


                                # --- AI Agent Interaction Section ---
                                st.subheader("Assistente de IA para Oportunidade")
                                # Removed the OpenAI client check here to always show the input area
                                # Add a placeholder message if client is not initialized
                                if st.session_state.get('ai_response') is None: # Check if session state has a response
                                 st.session_state['ai_response'] = "" # Initialize if not present

                                if client:
                                    user_query = st.text_area(f"Faça uma pergunta sobre a oportunidade {selected_opportunity_identifier}:", height=100, key='user_query')
                                    col_ai_button1, col_ai_button2 = st.columns(2)

                                    with col_ai_button1:
                                        if st.button("Obter Resposta da IA", use_container_width=True):
                                            if user_query:
                                                with st.spinner("Obtendo resposta da IA..."):
                                                    try:
                                                        # Construct the prompt
                                                        prompt = f"""
                                                        Você é um assistente de BI focado em analisar dados de oportunidades de negócios.
                                                        Sua tarefa é responder a perguntas sobre uma oportunidade específica com base nos dados fornecidos.
                                                        Seja conciso e útil, focando em insights de BI e na progressão da oportunidade.
                                                        **Use APENAS os dados fornecidos abaixo.**
                                                        Se a pergunta do usuário não puder ser respondida com os dados disponíveis, diga isso de forma educada.

                                                        Dados da Oportunidade com identificador {selected_opportunity_identifier}:

                                                        Detalhes Principais:
                                                        - ID: {opportunity_details.get('ID', 'N/A')}
                                                        - Título: {opportunity_details.get('Título', 'N/A')}
                                                        - Responsável: {opportunity_details.get('Responsável', 'N/A')}
                                                        - Estado: {opportunity_details.get('Estado', 'N/A')}
                                                        - Estágio Atual: {opportunity_details.get('Estágio', 'N/A')}
                                                        - Valor: R$ {opportunity_details.get('Valor', 'N/A')}
                                                        - Origem: {opportunity_details.get('Origem', 'N/A')}
                                                        - Prob %: {opportunity_details.get('Prob %', 'N/A')}
                                                        - OC: {opportunity_details.get('OC', 'N/A')}
                                                        - Data de Abertura: {opportunity_details.get('Data de abertura', 'N/A')}
                                                        - Data de Fechamento: {opportunity_details.get('Data fechamento', 'N/A')}

                                                        Detalhes de Fechamento (se aplicável):
                                                        - Valor Fechamento: R$ {opportunity_details.get('Valor fechamento', 'N/A')}
                                                        - Valor Rec. Fechamento: R$ {opportunity_details.get('Valor rec. fechamento', 'N/A')}
                                                        - Razão de Fechamento: {opportunity_details.get('Razão de fechamento', 'N/A')}
                                                        - Observação de Fechamento: {opportunity_details.get('Observação de fechamento', 'N/A')}

                                                        Linha do Tempo (Estágios e Tempos):
                                                        {opportunity_timeline[['Estágio', 'Data de abertura', 'Data fechamento', 'Time_in_Stage_Formatted']].to_string(index=False)}

                                                        Pergunta do Usuário: {user_query}

                                                        Responda em Português do Brasil.
                                                        """

                                                        response = client.chat.completions.create(
                                                            model="gpt-4o-mini", # Use a cost-effective model
                                                            messages=[
                                                                {"role": "system", "content": "Você é um assistente de BI útil e conciso."}, # System message for context
                                                                {"role": "user", "content": prompt}
                                                            ],
                                                            max_tokens=300 # Limit response length
                                                        )
                                                        st.session_state['ai_response'] = response.choices[0].message.content
                                                    except Exception as e:
                                                        st.session_state['ai_response'] = f"Erro ao comunicar com a API da OpenAI: {e}"
                                            else:
                                                st.session_state['ai_response'] = "Por favor, digite sua pergunta sobre a oportunidade."

                                    with col_ai_button2:
                                        if st.button("Limpar Resposta da IA", use_container_width=True):
                                            st.session_state['ai_response'] = "" # Clear the response

                                    # Display the AI response
                                    if st.session_state.get('ai_response'):
                                        st.text_area("Resposta da IA:", value=st.session_state['ai_response'], height=200, disabled=True, key='ai_response_display')
                                else:
                                    st.info("O assistente de IA está desabilitado porque a chave da API da OpenAI não foi configurada.")


                        except Exception as e:
                            st.error(f"Erro ao carregar detalhes da oportunidade {selected_opportunity_identifier}: {e}")

                except Exception as e:
                    st.error(f"Erro ao processar identificadores de oportunidade: {e}")


except Exception as e:
        st.error(f"Ocorreu um erro geral na aplicação: {e}")
        st.stop() # Stop the app execution on a critical error

