from math import pi
import numpy as np
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import plotly.graph_objects as go
from classes import PointCollection, Perceptron, MultiLayerPerceptron
from functions import create_figure, create_figure_mlp

st.set_page_config(
    page_title="Perceptron Visualization"
)


st.title("Perceptron Visualisation")
st.write(
    "Generate a dataset and observe how a single- or multi-layer perceptron seperates the two classes."
)

container_style = ["""
{
    background-color: #271d36;
    border-radius: 0.6em;
    padding: 0.5em;
}
""",
                   """
div {
    padding-right: 0.5rem
}
""",
                   """
div {
    padding-left: 0.1rem
}
"""]

# Save our PointCollection object in a session state
if "dataset" not in st.session_state:
    st.session_state.dataset = PointCollection()
# Save our Perceptron object in a session state
if "perceptron" not in st.session_state:
    st.session_state.perceptron = Perceptron()

# Init states for slider values
if "slider_value_x_loc" not in st.session_state:
    st.session_state.slider_value_x_loc = None
if "slider_value_y_loc" not in st.session_state:
    st.session_state.slider_value_y_loc = None
if "slider_value_scale" not in st.session_state:
    st.session_state.slider_value_scale = None
if "slider_value_rotation" not in st.session_state:
    st.session_state.slider_value_rotation = None
if "slider_value_label" not in st.session_state:
    st.session_state.slider_value_label = None

# Control the selected point collection to view/edit
if "collection_selection" not in st.session_state:
    st.session_state.collection_selection = "All"

# Control the stage of the app - Editing dataset or show perceptron
if "stage" not in st.session_state:
    st.session_state.stage = 0

if "add_edit" not in st.session_state:
    st.session_state.add_edit = 0

# Callback for the selectbox collection selection


def on_selectbox_change_collection_selection(selected):
    st.session_state.collection_selection = selected

# Callback functions to update PointCollection more smoothly


def on_slider_change_loc() -> None:
    st.session_state.dataset.change_collection_location(
        index=st.session_state.collection_selection,
        position_x=st.session_state.slider_value_x_loc,
        position_y=st.session_state.slider_value_y_loc)


def on_slider_change_scale() -> None:
    st.session_state.dataset.change_collection_scale(index=st.session_state.collection_selection,
                                                     scale=st.session_state.slider_value_scale)


def on_slider_change_rotation() -> None:
    st.session_state.dataset.change_collection_rotation(
        index=st.session_state.collection_selection,
        rotation_angle=st.session_state.slider_value_rotation*pi)


def on_slider_change_label() -> None:
    st.session_state.dataset.change_collection_label(
        index=st.session_state.collection_selection,
        label=st.session_state.slider_value_label
    )

# Callback for removing collection and exiting edit mode


def remove_collection(i: int):
    st.session_state.dataset.remove_collection(
        st.session_state.collection_selection)
    st.session_state.collection_selection = "All"
    st.session_state.add_edit = i

def remove_all_collections(i: int):
    total_collections = st.session_state.dataset.number_of_collections()
    if total_collections > 0:
        for _ in range(total_collections):
            st.session_state.dataset.remove_collection(0)
        st.session_state.add_edit = i


# Callback for controlling stage with buttons


def set_state(i: int) -> None:
    st.session_state.stage = i

# Callback for shifting between editing and adding point collections


def set_add_edit(i: int) -> None:
    st.session_state.add_edit = i


@st.cache_data
def pregen_datapoints():
    st.session_state.dataset.add_ellipse_collection(
        number_of_points=50, a=0.4, b=1, label=1)
    st.session_state.dataset.change_collection_rotation(
        index=0, rotation_angle=0.3*pi)
    st.session_state.dataset.change_collection_location(
        index=0, position_x=-1.84, position_y=0.98)
    st.session_state.dataset.change_collection_scale(index=0, scale=1.0)

    st.session_state.dataset.add_ellipse_collection(
        number_of_points=40, a=0.4, b=1, label=1)
    st.session_state.dataset.change_collection_rotation(
        index=1, rotation_angle=-0.2*pi)
    st.session_state.dataset.change_collection_location(
        index=1, position_x=-1.40, position_y=0.0)
    st.session_state.dataset.change_collection_scale(index=1, scale=0.66)

    st.session_state.dataset.add_ellipse_collection(
        number_of_points=50, a=0.4, b=1, label=-1)
    st.session_state.dataset.change_collection_rotation(
        index=2, rotation_angle=-0.2*pi)
    st.session_state.dataset.change_collection_location(
        index=2, position_x=-0.22, position_y=1.21)
    st.session_state.dataset.change_collection_scale(index=2, scale=1.0)

    st.session_state.dataset.add_ellipse_collection(
        number_of_points=40, a=0.4, b=1, label=-1)
    st.session_state.dataset.change_collection_rotation(
        index=3, rotation_angle=0.2*pi)
    st.session_state.dataset.change_collection_location(
        index=3, position_x=-0.40, position_y=0.1)
    st.session_state.dataset.change_collection_scale(index=3, scale=0.66)


pregen_datapoints()

col_left, col_middle, col_right = st.columns(3)

with col_left:
    st.button("Create / Edit Dataset", on_click=set_state,
              args=[1], use_container_width=True)

with col_middle:
    st.button("Perceptron Training", on_click=set_state,
              args=[2], use_container_width=True)
    
with col_right:
    st.button("Neural Network Training", on_click=set_state,
              args=[3], use_container_width=True)

st.markdown("""<hr style="height:6px;border:none;color:#262730;background-color:#262730;" /> """,
            unsafe_allow_html=True)

# editing mode
if st.session_state.stage == 1:
    col_left, col_right = st.columns(2)
    with col_left:
        col_inner_left, col_inner_right = st.columns(2)
        with col_inner_left:
            st.button("View/edit points", on_click=set_add_edit,
                      args=[1], use_container_width=True)
        with col_inner_right:
            st.button("Add new points", on_click=set_add_edit,
                      args=[2], use_container_width=True)
        if st.session_state.add_edit == 1:
            # select collection index
            collection_selections = ["All"]
            # collection_selections = []
            for index in range(st.session_state.dataset.number_of_collections()):
                collection_selections.append(index)
            selected_collection = st.selectbox(
                "Select collection", options=collection_selections, index=0)
            st.session_state.collection_selection = selected_collection

            # st.selectbox("Select collection", options=collection_selections, key="collection_selection")

            if st.session_state.collection_selection == "All":
                total_collections = st.session_state.dataset.number_of_collections()
                if total_collections > 0:
                    st.button("Remove all collections", use_container_width=True,
                        on_click=remove_all_collections, args=[4])
                else:
                    st.write("There are no point collections in the dataset. Press 'Add new points' to build a dataset.")

            else:
                # update states based on collection selection
                collection_loc = st.session_state.dataset.get_collection_location(
                    st.session_state.collection_selection)
                st.session_state.slider_value_x_loc = collection_loc[0]
                st.session_state.slider_value_y_loc = collection_loc[1]
                st.session_state.slider_value_scale = st.session_state.dataset.get_collection_scale(
                    st.session_state.collection_selection)
                st.session_state.slider_value_rotation = round(
                    st.session_state.dataset.get_collection_rotation(st.session_state.collection_selection)/pi, 1)
                st.session_state.slider_value_label = st.session_state.dataset.get_collection_label(
                    st.session_state.collection_selection)
                # location
                with stylable_container(key="location_sliders",
                                        css_styles=["""
                                        {
                                            background-color: #262730;
                                            border-radius: 0.6em;
                                            padding: 0.5em;
                                        }
                                        """,
                                                        """
                                        div {
                                            padding-right: 0.5rem
                                        }
                                        """,
                                                        """
                                        div {
                                            padding-left: 0.1rem
                                        }
                                        """]):
                    st.slider("X",
                            min_value=-5.0,
                            max_value=5.0,
                            key="slider_value_x_loc",
                            on_change=on_slider_change_loc)
                    st.slider("Y",
                            min_value=-5.0,
                            max_value=5.0,
                            key="slider_value_y_loc",
                            on_change=on_slider_change_loc)
                # scale
                with stylable_container(key="scale_slider",
                                        css_styles=["""
                                        {
                                            background-color: #262730;
                                            border-radius: 0.6em;
                                            padding: 0.5em;
                                        }
                                        """,
                                                        """
                                        div {
                                            padding-right: 0.5rem
                                        }
                                        """,
                                                        """
                                        div {
                                            padding-left: 0.1rem
                                        }
                                        """]):
                    st.slider("Scale",
                            min_value=0.01,
                            max_value=3.0,
                            key="slider_value_scale",
                            on_change=on_slider_change_scale)
                # rotation
                with stylable_container(key="angle_slider",
                                        css_styles=["""
                                        {
                                            background-color: #262730;
                                            border-radius: 0.6em;
                                            padding: 0.5em;
                                        }
                                        """,
                                                        """
                                        div {
                                            padding-right: 0.5rem
                                        }
                                        """,
                                                        """
                                        div {
                                            padding-left: 0.1rem
                                        }
                                        """]):
                    st.slider("Angle",
                            min_value=-2.0,
                            max_value=2.0,
                            step=0.1,
                            key="slider_value_rotation",
                            format="%fπ",
                            on_change=on_slider_change_rotation)
                # label
                with stylable_container(key="label_slider",
                                        css_styles=["""
                                        {
                                            background-color: #262730;
                                            border-radius: 0.6em;
                                            padding: 0.5em;
                                        }
                                        """,
                                                        """
                                        div {
                                            padding-right: 0.5rem
                                        }
                                        """,
                                                        """
                                        div {
                                            padding-left: 0.1rem
                                        }
                                        """]):
                    st.select_slider("Class label",
                                    options=[-1, 1],
                                    key="slider_value_label",
                                    on_change=on_slider_change_label)
                # removal
                st.button("Remove collection", use_container_width=True,
                          on_click=remove_collection, args=[3])

        if st.session_state.add_edit == 2:
            add_options = ["Circle", "Square", "Ellipse", "Rectangle"]
            add_type = st.selectbox("Collection type", options=add_options)
            if add_type == "Circle":
                number_of_points = st.number_input(
                    "Number of points", min_value=10, max_value=100, step=10)
                if st.button("Add collection!", use_container_width=True):
                    st.session_state.dataset.add_ellipse_collection(
                        number_of_points)
            elif add_type == "Square":
                number_of_points = st.number_input(
                    "Number of points", min_value=10, max_value=100, step=10)
                if st.button("Add collection!", use_container_width=True):
                    st.session_state.dataset.add_rectangle_collection(
                        number_of_points)
            elif add_type == "Ellipse":
                number_of_points = st.number_input(
                    "Number of points", min_value=10, max_value=100, step=10)
                eccentricity = st.number_input(
                    "Eccentricity", min_value=0.1, max_value=1.0, step=0.1)
                if st.button("Add collection!", use_container_width=True):
                    st.session_state.dataset.add_ellipse_collection(
                        number_of_points, eccentricity)
            elif add_type == "Rectangle":
                number_of_points = st.number_input(
                    "Number of points", min_value=10, max_value=100, step=10)
                short_side = st.number_input(
                    "Short side length", min_value=0.1, max_value=1.0, step=0.1)
                if st.button("Add collection!", use_container_width=True):
                    st.session_state.dataset.add_rectangle_collection(
                        number_of_points, short_side)

        if st.session_state.add_edit == 3:
            st.write("Point collection successfully removed from the set")
        if st.session_state.add_edit == 4:
            st.write("All point collections successfully removed")

    with col_right:
        view_selection = st.selectbox("Select view", options=(
            "Full view", "Current selecion"), index=0)
        if st.session_state.collection_selection == "All" or view_selection == "Full view":
            df = st.session_state.dataset.build_dataframe()
            with stylable_container(key="full_view_plot",
                                    css_styles=["""
                                    .main-svg:nth-of-type(1) {
                                        border-radius: 0.6em;
                                        border-style: solid;
                                        border-width: 1px;
                                        border-color: #41444C;
                                    }""",
                                                """
                                    .main-svg:nth-of-type(2) {
                                        padding: 0.6em;
                                    }"""]):
                fig = go.Figure()
                for label_value in [-1, 1]:
                    mask = df["label"] == label_value
                    fig.add_trace(go.Scatter(
                        x=df[mask]["x"],
                        y=df[mask]["y"],
                        mode="markers",
                        marker=dict(color="red" if label_value == -1 else "blue"),
                        name=f"Label {label_value}"
                    ))

                # highlight editing collection
                if not st.session_state.collection_selection == "All":
                    df = st.session_state.dataset.build_single_collection_dataframe(
                        st.session_state.collection_selection)
                    for label_value in [-1, 1]:
                        mask = df["label"] == label_value
                        fig.add_trace(go.Scatter(
                            x=df[mask]["x"],
                            y=df[mask]["y"],
                            mode="markers",
                            marker=dict(
                                color="rgba(31, 119, 180, 0.5)" if label_value == -1 else "rgba(255, 127, 14, 0.3)"),
                            name=f"Label {label_value}"
                        ))
                fig.update_layout(showlegend=False)
                fig.update_yaxes(scaleanchor="x",
                                scaleratio=1
                                )
                fig.update_layout(paper_bgcolor="#131720")
                fig.update_layout(plot_bgcolor="#131720")
                st.plotly_chart(fig)
        else:
            df = st.session_state.dataset.build_single_collection_dataframe(
                st.session_state.collection_selection)
            with stylable_container(key="selection_view_plot",
                                    css_styles=["""
                                    .main-svg:nth-of-type(1) {
                                        border-radius: 0.6em;
                                        border-style: solid;
                                        border-width: 1px;
                                        border-color: #41444C;
                                    }""",
                                                """
                                    .main-svg:nth-of-type(2) {
                                        padding: 0.6em;
                                    }"""]):
                fig = go.Figure()
                for label_value in [-1, 1]:
                    mask = df["label"] == label_value
                    fig.add_trace(go.Scatter(
                        x=df[mask]["x"],
                        y=df[mask]["y"],
                        mode="markers",
                        marker=dict(color="red" if label_value == -1 else "blue"),
                        name=f"Label {label_value}"
                    ))
                fig.update_yaxes(scaleanchor="x",
                                scaleratio=1
                                )
                fig.update_layout(paper_bgcolor="#131720")
                fig.update_layout(plot_bgcolor="#131720")
                st.plotly_chart(fig)

if st.session_state.stage == 2:
    total_collections = st.session_state.dataset.number_of_collections()
    if total_collections > 0:
        df = st.session_state.dataset.build_dataframe()

        x_train = np.array(df.drop(columns="label"))
        y_train = np.array(df["label"])

        epochs = 20

        w_history = st.session_state.perceptron.fit(
            x_train=x_train, y_train=y_train, epochs=epochs)

        with stylable_container(key="perceptron_visual",
                                css_styles=["""
                                .main-svg:nth-of-type(1) {
                                    border-radius: 0.6em;
                                    border-style: solid;
                                    border-width: 1px;
                                    border-color: #41444C;
                                }""",
                                            """
                                .main-svg:nth-of-type(2) {
                                    padding: 0.6em;
                                }"""]):
            # Create the base figure
            fig = create_figure(weights=w_history[0],
                                x_train=x_train, df=df)

            # Calculate the axis ranges
            x_min, x_max = x_train[:, 0].min() - 0.5, x_train[:, 0].max() + 0.5
            y_min, y_max = x_train[:, 1].min() - 0.5, x_train[:, 1].max() + 0.5

            # Update the layout
            fig.update_layout(
                title="                        Perceptron Decision Boundary",
                xaxis_title="X",
                yaxis_title="Y",
                width=800,
                height=600,
                xaxis={"range": [x_min, x_max], "visible": False,
                    "showticklabels": False, "hoverformat": ".2f"},
                yaxis={"range": [y_min, y_max], "visible": False,
                    "showticklabels": False, "hoverformat": ".2f"},
            )

            fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
            )

            # Create and add frames
            frames = [go.Frame(data=create_figure(weights=w, x_train=x_train, df=df).data, name=str(i))
                    for i, w in enumerate(w_history)]
            fig.frames = frames

            # Add slider and play button
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(label="Play",
                                method="animate",
                                args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
                            dict(label="Pause",
                                method="animate",
                                args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}])
                        ]
                    )
                ],
                sliders=[
                    dict(
                        steps=[
                            dict(
                                method="animate",
                                args=[[str(i)], {"frame": {"duration": 100,
                                                        "redraw": True}, "mode": "immediate"}],
                                label=str(i+1)
                            )
                            for i in range(epochs)
                        ],
                        transition={"duration": 0},
                        x=0,
                        y=0,
                        currentvalue={"font": {"size": 12}, "prefix": "Iteration: ",
                                    "visible": True, "xanchor": "center"},
                        len=0.9,
                    )
                ]
            )
            fig.update_layout(paper_bgcolor="#131720")
            fig.update_layout(plot_bgcolor="#131720")
            st.plotly_chart(fig)
            st.write("")

        with stylable_container(key="perceptron_explanation",
                                css_styles=["""
                                {
                                    background-color: #262730;
                                    border-radius: 0.6em;
                                    padding: 0.5em;
                                }
                                """,
                                            """
                                div {
                                    padding-right: 0.5rem
                                }
                                """,
                                            """
                                div {
                                    padding-left: 0.1rem
                                }
                                """]):
            st.write("The single-layer perceptron model")
            st.write("A single-layer perceptron is a simple type of artificial neuron used for binary classification, which means it can separate data into two categories. It takes several input values, each with an assigned weight (indicating importance), combines them, and outputs a classification.")
            st.write("Here's how it works:")
            st.markdown("- Weighted Sum: The perceptron calculates a weighted sum of the input values. Each input is multiplied by its weight, and these products are added together.")
            st.markdown("- Decision Boundary: The weights define a decision boundary, which is essentially a line (or hyperplane in higher dimensions) that separates the two classes. Points on one side of the boundary belong to one class, while points on the other side belong to the other class.")
            st.markdown("- Activation (Threshold): This weighted sum is passed through an activation function, typically a threshold function. If the sum is above the threshold, the perceptron outputs one class (e.g., '1'); if it's below, it outputs the other class (e.g., '0').")
            st.markdown("- Learning: During training, the perceptron adjusts its weights to improve classification accuracy, which moves the decision boundary to better separate the classes.")
            st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                list-style-position: inside;
            }
            </style>
            ''', unsafe_allow_html=True)
            st.write("In short, a single-layer perceptron decides which class an input belongs to by comparing the weighted sum to a threshold, and it uses its weights to define a decision boundary that separates the two classes. However, it can only solve problems that can be separated linearly (by a straight line in 2D or a flat plane in higher dimensions).")
    else:
        with stylable_container(key="single_set_empty",
                                css_styles=["""
                                {
                                    background-color: #262730;
                                    border-radius: 0.6em;
                                    padding: 0.5em;
                                }
                                """,
                                            """
                                div {
                                    padding-right: 0.5rem
                                }
                                """,
                                            """
                                div {
                                    padding-left: 0.1rem
                                }
                                """]):
            st.write("The dataset is currently empty.")
            st.write("To proceed with perceptron training you must add at least one point collection to your dataset.")
            st.write("-> Create/edit dataset -> Add new points")


if st.session_state.stage == 3:
    total_collections = st.session_state.dataset.number_of_collections()
    if total_collections > 0:
        df = st.session_state.dataset.build_dataframe()

        # class -1 to 0 because it was the better choice for mlp due to ReLU and sigmoid functions.
        df["label"] = df["label"].apply(lambda x: 0 if x == -1 else x)

        x_train = np.array(df.drop(columns="label"))
        y_train = np.array(df["label"])

        df["label"] = df["label"].astype(str)

        epochs = 101
        learning_rate = 0.002
        hidden_layers=[10,3]

        st.write("Hello")

        mlp = MultiLayerPerceptron(learning_rate=learning_rate, epochs=epochs, hidden_layers=hidden_layers)
        mse = mlp.fit(x_train, y_train)
        pred = mlp.predict(x_train)
        #print(np.sum(np.abs(y_train-pred)))

        with stylable_container(key="multi_perceptron_visual",
                                css_styles=["""
                                .main-svg:nth-of-type(1) {
                                    border-radius: 0.6em;
                                    border-style: solid;
                                    border-width: 1px;
                                    border-color: #41444C;
                                }""",
                                            """
                                .main-svg:nth-of-type(2) {
                                    padding: 0.6em;
                                }"""]):
            # Create the base figure
            fig = go.Figure()
            # dummy trace. Added because of a bug in plotly frames which removes the first trace.
            fig.add_trace(go.Scatter(
                x=df["x"][:1],
                y=df["y"][:1],
                mode="markers",
                marker={"color": "blue"},
                showlegend=False
                ))
            # add traces for the classes in our dataframe
            for class_value in ["0", "1"]:
                mask = df["label"] == class_value
                fig.add_trace(go.Scatter(
                    x=df[mask]["x"],
                    y=df[mask]["y"],
                    mode="markers",
                    marker=dict(color="red" if class_value == "0" else "blue"),
                    name=f"Class {class_value}        "
                ))

            # Calculate the axis ranges
            x_min, x_max = x_train[:, 0].min() - 0.5, x_train[:, 0].max() + 0.5
            y_min, y_max = x_train[:, 1].min() - 0.5, x_train[:, 1].max() + 0.5

            # Update the layout
            fig.update_layout(
                title="                        Neural Network Decision Boundary",
                xaxis_title="X",
                yaxis_title="Y",
                width=800,
                height=600,
                xaxis={"range": [x_min, x_max], "visible": False,
                    "showticklabels": False, "hoverformat": ".2f"},
                yaxis={"range": [y_min, y_max], "visible": False,
                    "showticklabels": False, "hoverformat": ".2f"},
            )

            fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
            )

            length = mlp.get_history_length()
            interval = mlp.get_save_interval()
            iterations = [i*interval for i in range(length-1)]
            iterations.append(epochs+1)

            # Create and add frames
            frames = [go.Frame(data=create_figure_mlp(model=mlp, features=x_train, index=i).data, name=str(i))
                    for i in range(length)]
            fig.frames = frames

            # Add slider and play button
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(label="Play",
                                method="animate",
                                args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
                            dict(label="Pause",
                                method="animate",
                                args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}])
                        ]
                    )
                ],
                sliders=[
                    dict(
                        steps=[
                            dict(
                                method="animate",
                                args=[[str(i)], {"frame": {"duration": 100,
                                                        "redraw": True}, "mode": "immediate"}],
                                label=str(iterations[i])
                            )
                            for i in range(length)
                        ],
                        transition={"duration": 0},
                        x=0,
                        y=0,
                        currentvalue={"font": {"size": 12}, "prefix": "Iteration: ",
                                    "visible": True, "xanchor": "center"},
                        len=0.9,
                    )
                ]
            )
            fig.update_layout(paper_bgcolor="#131720")
            fig.update_layout(plot_bgcolor="#131720")
            st.plotly_chart(fig)
            st.write("")

        with stylable_container(key="multi_perceptron_explanation",
                                css_styles=["""
                                {
                                    background-color: #262730;
                                    border-radius: 0.6em;
                                    padding: 0.5em;
                                }
                                """,
                                            """
                                div {
                                    padding-right: 0.5rem
                                }
                                """,
                                            """
                                div {
                                    padding-left: 0.1rem
                                }
                                """]):
            st.write("The multi-layer perceptron model")
            st.write("A multi-layer perceptron (MLP) builds on the single-layer perceptron by adding more layers, making it much more powerful and flexible. While a single-layer perceptron can only classify data that's linearly separable (like a line or flat plane dividing two classes), a multi-layer perceptron can tackle more complex problems with non-linear boundaries.")
            st.write("Here's how it works:")
            st.markdown("- Layers of Neurons: An MLP has an input layer, one or more hidden layers, and an output layer. Each layer contains several neurons, and each neuron in a layer is connected to every neuron in the next layer, creating a 'fully connected' network.")
            st.markdown("- Weighted Sum and Activation (Non-Linear): Each neuron in the MLP computes a weighted sum of its inputs, like in the single-layer perceptron. But here, instead of using a simple threshold, it typically applies a non-linear activation function (like ReLU or sigmoid). This non-linearity allows the MLP to learn complex relationships in the data.")
            st.markdown("- Learning through Backpropagation: MLPs use a process called backpropagation to learn. When the network makes a mistake, it calculates how much each neuron contributed to the error, then adjusts the weights to reduce the error. This process repeats for many iterations until the network learns to classify data accurately.")
            st.markdown("- Complex Decision Boundaries: Thanks to the hidden layers and non-linear activations, MLPs can create intricate decision boundaries that separate data in complex ways. Unlike the single-layer perceptron’s straight-line boundary, an MLP can form curved boundaries that better fit real-world data.")
            st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                list-style-position: inside;
            }
            </style>
            ''', unsafe_allow_html=True)
            st.write("In summary, while a single-layer perceptron is limited to problems that can be separated by a straight line, a multi-layer perceptron can classify data with complex, non-linear boundaries. It achieves this through multiple layers of neurons and non-linear activation functions, learning to adapt its weights in a process called backpropagation.")
    else:
        with stylable_container(key="multi_set_empty",
                                css_styles=["""
                                {
                                    background-color: #262730;
                                    border-radius: 0.6em;
                                    padding: 0.5em;
                                }
                                """,
                                            """
                                div {
                                    padding-right: 0.5rem
                                }
                                """,
                                            """
                                div {
                                    padding-left: 0.1rem
                                }
                                """]):
            st.write("The dataset is currently empty.")
            st.write("To proceed with neural network training you must add at least one point collection to your dataset.")
            st.write("-> Create/edit dataset -> Add new points")

with stylable_container(key="contact_info",
                        css_styles=["""
                        {
                            background-color: #262730;
                            border-radius: 0.6em;
                            padding: 0.5em;
                        }
                        """,
                                    """
                        div {
                            padding-right: 0.5rem
                        }
                        """,
                                    """
                        div {
                            padding-left: 0.1rem
                        }
                        """]):
    st.write("Created by: Robert Örneving")
    url = "https://www.streamlit.io"
    st.write("[LinkedIn](%s)" % url)
    url2 = "https://github.com/Eaglewing89/perceptron_visualization"
    st.write("[GitHub repository](%s)" % url2)