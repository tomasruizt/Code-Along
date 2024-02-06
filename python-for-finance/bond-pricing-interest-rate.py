import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Bond Pricing and Interest Rate")
st.markdown(
    """
    This app calculates the price of a bond and how it changes with the interest rate.

    Assume the bond has a face value of $100, a coupon rate of 3%. 
    The maturity and the discount rate can be adjusted.
    As you increase the discount rate, the bond price decreases.
    The effect is magnified by longer maturities.
    """
)

with st.echo():
    # Define the bond parameters
    face_value = 100
    coupon_rate = 0.03  # 3%

    def bond_price(maturity: int, discount_rate: float) -> float:
        T = maturity
        r = discount_rate
        # d stands for discounted
        d_face_value = face_value / (1 + r) ** T
        d_coupons = face_value * coupon_rate * (1 - (1 + r) ** -T) / r
        return d_face_value + d_coupons

maturity = st.slider("Maturity (years)", min_value=1, max_value=30, value=15)
discount_rate = st.slider("Discount rate [%]", min_value=0.0, max_value=6.0, value=coupon_rate * 100, step=0.5) / 100

st.write("##### The bond price is", round(bond_price(maturity, discount_rate), 2))
st.markdown(f"An intersting observation: If the discount rate (currently {discount_rate:.2f}) is equal to the coupon rate ({coupon_rate:.2f}), the bond price is equal to the face value, no matter the maturity.")

st.markdown(
    """
    ## Sensitivity to interest rate
    Longer maturities are more sensitive to interest rate changes.
    Let us illustrate this by plotting the bond price as a function of hikes in the interest rate for different maturities.
    """
)



# Create a range of interest rate increases
increases = np.linspace(-coupon_rate + 0.001, 0.06, 100)
discount_rates = coupon_rate + increases
maturities = np.array([1, 5, 10, 20, 30])
prices = np.array([bond_price(maturity, discount_rates) for maturity in maturities])

fig, ax = plt.subplots()
ax.set_title("Bond price sensitivity to interest rate")
ax.set_xlabel("Increase in interest rate [%]")
ax.set_ylabel("Bond price")
ax.grid(True)
# plot the curves with viridis color map, and add a legend
for i, maturity in enumerate(maturities):
    ax.plot(increases * 100, prices[i], label=f"Maturity {maturity} years", color=plt.cm.viridis(i / len(maturities)))
ax.legend()
st.pyplot(fig)



