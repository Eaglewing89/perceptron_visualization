from math import pi
import streamlit as st
import plotly.graph_objects as go
from classes import PointCollection


st.set_page_config(
    page_title="Interactive Perceptron model", page_icon=":pie_chart"
)

# Save our PointCollection object in a session state
if "dataset" not in st.session_state:
    st.session_state.dataset = PointCollection()

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
    st.session_state.dataset.remove_collection(st.session_state.collection_selection)
    st.session_state.collection_selection = "All"
    st.session_state.add_edit = i

# Callback for controlling stage with buttons
def set_state(i: int) -> None:
    st.session_state.stage = i

# Callback for shifting between editing and adding point collections
def set_add_edit(i: int) -> None:
    st.session_state.add_edit = i


st.title("Perceptron Visualisation")
st.write(
    "Create points collections and see how a single layer perceptron manages to seperate the two classes."
)

# st.header("Generate dataset")


@st.cache_data
def pregen_datapoints():
    st.session_state.dataset.add_ellipse_collection(
        number_of_points=50, a=2, b=1, label=-1)
    st.session_state.dataset.change_collection_rotation(
        index=0, rotation_angle=0.2*pi)
    st.session_state.dataset.change_collection_location(
        index=0, position_x=1.75, position_y=2)
    st.session_state.dataset.change_collection_scale(index=0, scale=1.1)

    st.session_state.dataset.add_ellipse_collection(
        number_of_points=50, a=2, b=1, label=1)
    st.session_state.dataset.change_collection_rotation(
        index=1, rotation_angle=-0.2*pi)
    st.session_state.dataset.change_collection_location(
        index=1, position_x=-1.75, position_y=-1)
    st.session_state.dataset.change_collection_scale(index=1, scale=0.1)


pregen_datapoints()

col_left, col_right = st.columns(2)

with col_left:
    st.button("Create / Edit Dataset", on_click=set_state, args=[1], use_container_width=True)

with col_right:
    st.button("Perceptron Training", on_click=set_state, args=[2], use_container_width=True)

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

            #st.selectbox("Select collection", options=collection_selections, key="collection_selection")            

            if st.session_state.collection_selection == "All":
                st.write("Editing 'All' not yet implemented")
                st.write(st.session_state.collection_selection)

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
                st.slider("Scale",
                          min_value=0.01,
                          max_value=2.0,
                          key="slider_value_scale",
                          on_change=on_slider_change_scale)
                # rotation
                st.slider("Angle",
                          min_value=-2.0,
                          max_value=2.0,
                          step=0.1,
                          key="slider_value_rotation",
                          format="%fπ",
                          on_change=on_slider_change_rotation)
                # label
                st.select_slider("Class label",
                                 options=[-1, 1],
                                 key="slider_value_label",
                                 on_change=on_slider_change_label)
                # removal
                st.button("Remove collection", use_container_width=True, on_click=remove_collection, args=[3])

        if st.session_state.add_edit == 2:
            add_options = ["Circle", "Square", "Ellipse"]
            add_type = st.selectbox("Collection type", options=add_options)
            if add_type == "Circle":
                number_of_points = st.number_input("Number of points", min_value=10, max_value=100, step=10)
                if st.button("Add collection!",use_container_width=True):
                    st.session_state.dataset.add_ellipse_collection(number_of_points)
            elif add_type == "Square":
                st.write("square here")
            elif add_type == "Ellipse":
                st.write("ellipse here")
                number_of_points = st.number_input("Number of points", min_value=10, max_value=100, step=10)
                eccentricity = st.number_input("Eccentricity", min_value=0.1, max_value=2.0, step=0.1)
                if st.button("Add collection!",use_container_width=True):
                    st.session_state.dataset.add_ellipse_collection(number_of_points, eccentricity)
        if st.session_state.add_edit == 3:
            st.write("Point collection successfully removed from the set")

    with col_right:
        view_selection = st.selectbox("Select view", options=(
            "Full view", "Current selecion"), index=0)
        if st.session_state.collection_selection == "All" or view_selection == "Full view":
            df = st.session_state.dataset.build_dataframe()
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
                df = st.session_state.dataset.build_single_collection_dataframe(st.session_state.collection_selection)
                for label_value in [-1, 1]:
                    mask = df["label"] == label_value
                    fig.add_trace(go.Scatter(
                        x=df[mask]["x"],
                        y=df[mask]["y"],
                        mode="markers",
                        marker=dict(color="rgba(31, 119, 180, 0.5)" if label_value == -1 else "rgba(255, 127, 14, 0.3)"),
                        name=f"Label {label_value}"
                    ))
# color_discrete_map = {-1: 'rgb(31, 119, 180)', 1: 'rgb(255, 127, 14)'}

            fig.update_layout(showlegend=False)
            fig.update_yaxes(scaleanchor="x",
                             scaleratio=1
                             )
            st.plotly_chart(fig)
        else:
            df = st.session_state.dataset.build_single_collection_dataframe(
                st.session_state.collection_selection)
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
            st.plotly_chart(fig)

if st.session_state.stage == 2:

    st.write("perceptron training here")

# selections = ["All"]
# if dataset.collections_points:
#     for index, _ in enumerate(dataset.collections_scales):
#         selections.append(index)
# option = st.selectbox("Select collection of points", selections, index=0)

# option = st.selectbox(
#         "Select view",
#         ("Full view", "Current editable collection"), index=0)

# if selections == "All":
#     df = dataset.build_dataframe()

# fig = go.Figure()

# for class_value in [-1, 1]:
#     mask = df["label"] == class_value
#     fig.add_trace(go.Scatter(
#         x=df[mask]['x'],
#         y=df[mask]['y'],
#         mode='markers',
#         marker=dict(color='red' if class_value == -1 else 'blue'),
#         name=f'Class {class_value}'
#     ))
