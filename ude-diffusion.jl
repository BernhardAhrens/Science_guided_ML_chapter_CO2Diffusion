cd("/User/homes/bahrens/Science_guided_ML_chapter_CO2Diffusion")

# load Project.toml and install packages
using Pkg; Pkg.activate("."); Pkg.instantiate()

using DiffEqFlux, Optim, DiffEqSensitivity, DiffEqOperators, Flux, LinearAlgebra, Interpolations, Plots, Dates, DataFrames, DataFramesMeta, CSV, StatsBase

file = "Synthetic4BookChap.csv"

dfall = CSV.read(file, DataFrame, normalizenames=true, missingstring = "NA", dateformat="yyyy-mm-ddTHH:MM:SSZ")

dfall

println(names(dfall))

dropmissing!(dfall)

subdfall = dfall[:,["siteID", "DateTime", "wdefCum", "H2Ostor1", "H2Ostor2", "Drainage", "SW_POT_sm", "SW_POT_sm_diff", "RUE_syn", "GPP_syn", "Rb_syn", "RECO_syn", "NEE_syn", "GPP_DT", "GPP_NT", "H_CORR", "H", "H_QC", "H_RANDUNC", "H_RANDUNC_N", "LE_CORR", "LE", "LE_QC", "LE_RANDUNC", "LE_RANDUNC_N", "NEE", "NETRAD", "P", "PA", "PPFD_DIF", "PPFD_IN", "PPFD_OUT", "RECO_DT", "RECO_NT", "SWC_1", "SWC_2", "SW_IN", "SW_IN_QC", "SW_IN_POT", "TA", "TS_1", "TS_2", "VPD", "WS"]]

subdfall = @linq subdfall |>
  transform(yearmonth = DateTime.(Year.(:DateTime),Month.(:DateTime))) |>
  transform(yearmonthday = DateTime.(Year.(:DateTime),Month.(:DateTime),Day.(:DateTime))) |>
  transform(yearmonthday = DateTime.(Year.(:DateTime),Month.(:DateTime),Day.(:DateTime))) |>
  transform(yearmonthdayhour = DateTime.(Year.(:DateTime),Month.(:DateTime),Day.(:DateTime),Hour.(:DateTime))) |>
  groupby([:siteID,:yearmonth])

using Statistics

dfk = combine(subdfall, valuecols(subdfall)[2:end-3] .=> mean .=> valuecols(subdfall)[2:end-3])

#===============================================================================
## plotting
===============================================================================#
#Plots.scalefontsizes(1/1.6)
gr(size=(700*1.6,700), dpi = 600, thickness_scaling = 1.5)
plot(dfk.yearmonth,dfk.wdefCum)

for i in names(dfk)
    p1 = plot(dfk.yearmonth,dfk[:,i], title = i, label = "")
    png(p1, joinpath("png", i))
end

#===============================================================================
# # make it work with sciml_train
===============================================================================#
# https://github.com/SciML/DiffEqFlux.jl/issues/432
# for sciml_train

# implement minibatches https://diffeqflux.sciml.ai/v1.24/examples/minibatch/

# https://github.com/SciML/DiffEqFlux.jl/issues/432

fractionofday(t) = (t-DateTime(year(t),month(t),day(t)))/Millisecond(1)/(1000*60*60)/24

fractional_dayofyear(t) = dayofyear(t)+fractionofday(t)

fractionofyear(t) = ifelse(isleapyear(t),fractional_dayofyear(t)/365,fractional_dayofyear(t)/366)

fractional_year(t) = fractionofyear(t) + year(t)

dfk.dt = dfk.yearmonth

dfk = @linq dfk |>
  transform(Year = string.(year.(:dt))) |>
  #where(:Year .== "2013") |>
  transform(dt_wo_year = :dt .- Year.(:dt)) |>
  transform(day_of_year = fractional_dayofyear.(:dt)) |>
  transform(fractional_year = fractional_year.(:dt)) |>
  transform(timestep_one = cumsum([1;diff(:fractional_year)/minimum(diff(:fractional_year))])
  #transform(timestep_one = fractional_year.(:dt)
  )

show(dfk, summary = false)

#  make it O(1) memory?
#dfk.timestep_one = dfk.timestep_one / maximum(dfk.timestep_one)

time_step_base = minimum(diff(dfk.timestep_one))

# we do it on yearly scale, should we change it?
#unit_conv_flux = Float32(60*60*24*365)
unit_conv_flux = Float32(60*60*0.5) 
#unit_conv_flux = Float32(60*60*24*30) 


#===============================================================================
## linear interpolated functions of forcing
===============================================================================#
# temperature and moisture input
# append first value for initial condition
itp_tsoil = LinearInterpolation(Float32[-3.0*maximum(dfk.timestep_one); dfk.timestep_one; 1.1*maximum(dfk.timestep_one)],Float32[mean(dfk.TA);dfk.TA;mean(dfk.TA)])
itp_swc = LinearInterpolation(Float32[-3.0*maximum(dfk.timestep_one); dfk.timestep_one; 1.1*maximum(dfk.timestep_one)],Float32[mean(dfk.SWC_1);dfk.SWC_1;mean(dfk.SWC_1)]/100)

swc_min = Float32.(minimum(dfk.SWC_1/100))
swc_max = Float32.(maximum(dfk.SWC_1/100))

# assume maximum soil moisture plus something
porosity = Float32.(maximum(dfk.SWC_1) + 1e-3)/100

#===============================================================================
## DAMM function as creator of respiration flux
===============================================================================#
function fDAMM(Temp,Moist;p)
  Rb,Q10,KmSx,KmO2 = p

  O2airfrac = Float32(0.209) # volume fraction of O2 in air
  Sxtot = Float32(0.048) # The total amount of the substrate Sx in the bulk soil
  psx = Float32(4.14e-4) # fraction of Sxtot that is soluble
  Dliq = Float32(3.17) # diffusion coefficient of the substrate in liquid phase
  Dgas = Float32(1.67) # Dgas is a diffusion coefficient for O2 in air
  
  Temp_ref = Float32(10.0) # reference temperature

  Sx = Sxtot*psx*Dliq*Moist.^Float32(3) 
  O2 = Dgas*O2airfrac*((porosity .- Moist).^(Float32(4.0)/Float32(3.0))) 
  MMSx = Sx ./ (KmSx .+ Sx)
  MMO2 = O2 ./ (KmO2 .+ O2)

  R = Rb .* Q10.^((Temp .- Temp_ref) ./ Float32(10.0)) #.* MMSx .* MMO2
end

# guessed parameters Rb,Q10. KmSx and KmO2 as in Davidson 2020
# 1kg = 83.2591 mol 
M_C = 12.0107e03 # kg mol-1
aSx = 5.38e10 # kg m-3 h-1
aSx_time_step = aSx/2 # kg m-3 0.5h-1

Rb = maximum(skipmissing(dfk.Rb_syn)) # mol m-3 0.5h-1

pDAMM = Float32.([Rb,1.5,9.97e-7,0.121])

resDAMM = fDAMM(dfk.TA,dfk.SWC_1/100;p=pDAMM)

plot(dfk.timestep_one, resDAMM)

#===============================================================================
## conversions from Eliza
===============================================================================#
# takes alt in m and gives p in mbar
function press_alt(x)
  (101325.f0*(1.f0 - 2.25577f0*10f0^-5.0f0*x)^5.25588f0)/100.0f0
end 
press_alt(1820)*100

# Eliza conversion
# function for air molar volume in mol/m3, adjusted for pressure and T
function rho_air(temp,alt=970.f0) # 970 m http://sites.fluxdata.org/AT-Neu/
  1.f0/22.4f0*(press_alt(alt)/1013.f0)/(273.15f0/(temp+273.15f0))*1000f0
  # air density in mol/m3, adjusted for pressure and T
end

28.97*rho_air(0) # g/mol* mol/m3 = 1.xx g/m3 https://en.wikipedia.org/wiki/Density_of_air#:~:text=This%20quantity%20may%20vary%20slightly,of%201.2041%20kg%2Fm3.

# CO2 ??mol/mol = CO2 in mol m-3 / rho_air in mol m-3
function umolcubic_to_ppm(umolcubic,temp)
  umolcubic ./ rho_air.(temp) #./ porosity #(porosity .- moist) ./ (porosity .- moist)
end

# CO2 in mol m-3 = CO2 ??mol/mol * rho_air in mol m-3
function ppm_to_umolcubic(ppm,temp)
  ppm .* rho_air.(temp) #./ porosity #(porosity .- moist) ./ (porosity .- moist)
end

#===============================================================================
## Diffusion coefficients
===============================================================================#

# CO2 diffusion coefficient in free air
f_Da = function(Temp; Pressure=81288.f0)
  Da0 = Float32(1.47e-05)
  T0 = Float32(293.15)
  Pressure0 = Float32(1.013e05)
  Temp = Temp + Float32(273.15)
  Da0*((Temp/T0)^(Float32(1.75)))*(Pressure/Pressure0)
end

# tortuosity function
f_tortuosity = function(Moist; Ds_method)
  ??a = porosity .- Moist
  if Ds_method == "Penman" 
    0.66f0*??a
  elseif Ds_method == "Marshall" 
    ??a.^1.5f0
  elseif Ds_method == "Millington_Quirk" 
    ??a.^(10.0f0/3.0f0)/porosity^2.0f0 # hmm Millington Quirk accroding to Herbst is ??a^(7.0/3.0) / porosity * ??a = ??a^(10.0/3.0) / porosity but Simunek does / porosity^2
  elseif Ds_method == "Moldrup" 
    porosity^2.0f0*(??a/porosity).^(2.9f0*0.8f0) 
    # 0.8 is the percentage of mineral soil with particle size >2??m; 2.9 is a constant
  else
    println("Ds_method not defined, should be one of Penman, Marshall, Millington_Quirk, Moldrup")
  end
end

# 
f_Ds = function(Temp,Moist; Ds_method = "Penman", Pressure=1.013e05)
  f_Da(Temp; Pressure = Pressure)*f_tortuosity(Moist; Ds_method = Ds_method)
end

t = Float32.(dfk.timestep_one)
Ds_method = "Penman"
v_tortuosity = f_tortuosity(itp_swc(t); Ds_method = Ds_method) #* unit_conv_flux
plot(v_tortuosity, label = "$Ds_method tortuosity", width = 1.5)

Ds_method = "Marshall"
v_tortuosity = f_tortuosity(itp_swc(t); Ds_method = Ds_method) #* unit_conv_flux
plot!(v_tortuosity, label = "$Ds_method tortuosity", width = 1.5)

Ds_method = "Millington_Quirk"
v_tortuosity = f_tortuosity(itp_swc(t); Ds_method = Ds_method) #* unit_conv_flux
plot!(v_tortuosity, label = "$Ds_method tortuosity", width = 1.5)

Ds_method = "Moldrup"
v_tortuosity = f_tortuosity(itp_swc(t); Ds_method = Ds_method) #* unit_conv_flux
plot!(v_tortuosity, label = "$Ds_method tortuosity", legend = :inline, width = 1.5, xlab = "Months", ylab = "??")
using Plots.PlotMeasures: mm
# Plots.scalefontsizes(1/1.6)

myplot = function(filename)
  savefig("$filename.png")
  savefig("$filename.svg")
end

myplot("tortuosity")

#===============================================================================
# learn Ds for pretraining
===============================================================================#

# explanatory variables
x_dfk = @linq dfk |> 
  select(#:par_unknown,
  #,:windspeed_d1_ms_1
  #:winddir_deg,:precip_mm,
  #:tair_degc,:precip_mm,
  :SWC_1
  #:par_unknown, :windspeed_d1_ms_1
  #:soilco2_10cm_ppm_PLOT,:soilco2_20cm_ppm_PLOT
  )
x_in = Array(x_dfk)'

# normalization
sdf = fit(UnitRangeTransform, x_in, dims=2)
x_in = (Float32.(StatsBase.transform(sdf, x_in)))*2 .- 1.f0
plot(x_in')

# Z-normalization
#x_in2 = Flux.normalise(Array(x_dfk)')
#plot!(x_in2')

#L = FastChain(FastDense(size(x_in,1), 16, tanh),FastDense(16, 16, tanh),FastDense(16, 1, x -> x^2))
L = FastChain(FastDense(size(x_in,1), 8, gelu),FastDense(8, 8, gelu),FastDense(8, 8, gelu),FastDense(8, 8, gelu),FastDense(8, 8, gelu),FastDense(8, 1, x -> x^2))
p_random = Float32.(initial_params(L))

L2(x) = sum(abs2, x)/length(x)
function loss_simple(??)
  mod = L(x_in,??)'
  Flux.mse(mod,v_tortuosity) + 0.01*L2(??)
end

loss_simple(p_random)

plot(v_tortuosity)
plot!(L(x_in,p_random)')

losses = []
callback(??,l) = begin
  push!(losses, l)
  if length(losses)%10==0
      println("Current loss after $(length(losses)) iterations: $(losses[end])")
  end
  false
end
outsideDs = DiffEqFlux.sciml_train(loss_simple, p_random, BFGS(initial_stepnorm = 0.001), maxiters = 1000, cb = callback)

plot(v_tortuosity)
plot!(L(x_in,outsideDs.minimizer)')


#===============================================================================
# setup PDE
===============================================================================#
N??? = 4
??x = 0.05f0 # in m
knots = range(??x, step=??x, length=N???)

println(knots)

scatter(knots, labels = "", title = "Mid points")
xlabel!("Index")
ord_deriv = 2
ord_approx = 2

# define the diffusion term
????? = CenteredDifference(ord_deriv, ord_approx, ??x, N???)


#===============================================================================
# PDE 
===============================================================================#
DepthRb = exp.(-1.f0/0.1f0*knots)

gr(size=(800,700))
#Plots.scalefontsizes(2)
plot(DepthRb,knots, ylabel = "Depth", label = "")
yaxis!(:flip)

DepthRb = DepthRb*length(knots)/sum(DepthRb)
Rbfactor = Float32.(ones(length(knots)))
Rbfactor = Rbfactor/(N???*??x)#.*DepthRb# m
#Rbfactor[3:22] .= 0.0

DepthDecline = Float32.(ones(N???)) #exp.(-1/0.5*knots)
DepthDecline = DepthDecline*maximum(DepthDecline)
gr(size=(800,700))
#Plots.scalefontsizes(2)
plot(DepthDecline,knots, ylabel = "Depth", label = "")
yaxis!(:flip)

saveat = Float32.(dfk.timestep_one) 

bc = NeumannBC((0.0f0,0.0f0),??x,1)

xmid = 0.05f0

# https://www.stochasticlifestyle.com/solving-systems-stochastic-pdes-using-gpus-julia/
# zero-flux Neumann Chris Rackauckas
# try to get DAMM back to check HybridML
function f(du,u,p,t)

  tsoil = itp_tsoil(t)
  swc = itp_swc(t)

  p_Ds = f_Ds.(tsoil,swc; Ds_method = Ds_method) .* DepthDecline * unit_conv_flux # multiply by unit_conv_flux to go from per second to per halfhour
  gradient_boundary = (u[2] - ppm_to_umolcubic(2092.83,tsoil[1]))/(0.5*??x) # mean(dfk.soilco2_0cm_ppm_PLOT)
  #gradient_boundary = (u[3] - u[2])/(??x^2.0) # repeated flux mean(dfk.soilco2_0cm_ppm_PLOT)
  efflux = gradient_boundary * f_Da(tsoil[1]) * unit_conv_flux
  #gamma = gradient_boundary/(f_Da(itp_tsoil(t))*unit_conv_flux)
  #bc = NeumannBC((gamma,0.0),??x,1)

  u_cnn_1   = [(u[3] - u[2])/((??x))]
  u_cnn     = [(u[i-1] - 2 * u[i] + u[i+1])/(??x) for i in 3:N???]
  u_cnn_end = [(u[end-1] - u[end])/((??x))]
  dDiff_c = (p_Ds.*vcat(u_cnn_1, u_cnn, u_cnn_end))/??x

  Resp = Rbfactor.*(fDAMM.(tsoil,swc;p=pDAMM) * ??x) * unit_conv_flux #./ (porosity .- swc)

  du[2:N???+1] = dDiff_c .+ Resp     #.- u[1:N???] #.+ dDiff_u
  du[2] = du[2] - efflux
  du[1] = efflux - u[1]*unit_conv_flux

end

u0 = Float32.(fill(0.0,N???+1))

p = Float32(0.1)

# extra time for spin-up 
tspan = Float32.((-3.0*maximum(saveat),maximum(saveat)))
#tspan = (minimum(saveat),maximum(saveat))

using DifferentialEquations
problem = ODEProblem(f,vec(u0),tspan,p)

# Ds_methods
obs = dfk
function plot_Flux_Conc(solution)
  p1 = plot(solution[:,1], labels = "Q?????? model")
  plot!(resDAMM, labels = "DAMM")
  ppm = umolcubic_to_ppm(solution[:,[1,2,4].+1],obs.TA)

  # pick depth, unit check???
  pmod = plot(ppm,
   #title = "Q?????? model + Diffusion $Ds_method modelled CO??? (ppm)"
   title = "CO??? (ppm)"
   #,labels = ["5 cm" "10 cm" "20 cm"], legend = :inline)
   ,labels = ["mod 5 cm" "mod 10 cm" "mod 20 cm"], legend = :inline)
  # http://painterqubits.github.io/Unitful.jl/stable/ would be cool

  plot(pmod, p1, width = 1, layout = (2,1))
  plot!(size=(1100,700))
end

function plot_Flux(solution)
  p1 = plot(solution[:,1], labels = "Q?????? model + Diffusion $Ds_method", colour = 4, legend = :topright, title = "CO??? efflux (??mol m?? s?????)", xlab = "Months", width = 3)
  plot!(resDAMM, labels = "Q?????? model" , colour = 5, width = 3)
  plot!(size=(1100,700), left_margin=5mm, bottom_margin=5mm)
end

gr()
tuple_Ds_methods = ("Penman", "Marshall","Millington_Quirk","Moldrup")
for i in tuple_Ds_methods
  Ds_method = i
  @time solution = Array(solve(problem,TRBDF2(autodiff=false),reltol=1e-5,abstol=1e-5, saveat = saveat))'
  display(plot_Flux(solution))
  display(plot_Flux_Conc(solution))
  plot_Flux
end


@time solution = Array(solve(problem,TRBDF2(autodiff=false),reltol=1e-5,abstol=1e-5, saveat = saveat))'


gr(size=(700*1.6,700), dpi = 600, thickness_scaling = 1.5)
plot_Flux(solution)
myplot("Q10_vs_Q10_Diffusion")

# make artificial data
artificial = deepcopy(obs)

artificial.co2flux_umol_m2_s1 = solution[:,1]
artificial.soilco2_5cm_ppm_PLOT = Float32.(umolcubic_to_ppm(solution[:,2],obs.TA))
artificial.soilco2_10cm_ppm_PLOT = Float32.(umolcubic_to_ppm(solution[:,3],obs.TA))
artificial.soilco2_20cm_ppm_PLOT = Float32.(umolcubic_to_ppm(solution[:,5],obs.TA))


#===============================================================================
# Okay, let's start with some calibration to the Observations
===============================================================================#
# very very good example:
# https://discourse.julialang.org/t/galacticoptim-trips-over-boundary-conditions-in-diffeqoperators/55060
# https://discourse.julialang.org/t/obscure-multiplication-Ds_method-error-when-using-sciml-train/45906

# should be out-of-place apparently
# https://github.com/SciML/DiffEqFlux.jl/issues/413

#L_itp = FastChain((x,p) -> [itp_swc(x),itp_tsoil(x)],L)
pML = Float32.(outsideDs.minimizer) #Float32.(initial_params(L)) #Float32.(really.minimizer) #
#u0 = zeros(N???+1)
# try to get DAMM back to check HybridML

function fp(du,u,p,t)
    tsoil = itp_tsoil(t)
    swc = itp_swc(t)

    #tsoil_nn = (tsoil - tsoil_min)/(tsoil_max - tsoil_min)
    swc_nn =   ((swc - swc_min)/(swc_max - swc_min))*2.f0 - 1.f0
  
    p_PBM = [exp(p[1]),exp(p[2]),pDAMM[3],pDAMM[4]]
    p_ML = p[3:end]

    p_Da = f_Da(tsoil; Pressure = 1.013e05) .* DepthDecline * unit_conv_flux
    p_tortuosity = L(swc_nn,p_ML)
    #p_tortuosity = f_tortuosity(swc; Ds_method = Ds_method)

    p_Ds = p_Da*p_tortuosity

    gradient_boundary = (u[2] - ppm_to_umolcubic(2092.83f0,tsoil[1]))/(0.5f0*??x) # mean(dfk.soilco2_0cm_ppm_PLOT)
    efflux = gradient_boundary * f_Da(tsoil[1]) * unit_conv_flux
  
    u_cnn_1   = [(u[3] - u[2])/((??x))]
    u_cnn     = [(u[i-1] - 2 * u[i] + u[i+1])/(??x) for i in 3:N???]
    u_cnn_end = [(u[end-1] - u[end])/((??x))]
    dDiff_c = (p_Ds.*vcat(u_cnn_1, u_cnn, u_cnn_end))/??x

    Resp = Rbfactor.*(fDAMM.(tsoil,swc;p=p_PBM) * ??x) * unit_conv_flux #./ (porosity .- swc)
  
    #du[:,1] = dF = dDiff_c .- F*unit_conv_flux # no accumulation in the first layer # cheating pretending its a flux. Problem is that I have not figured out a way to return terms from a DifferentialEquations and put it through the Automatic Differentiation procedure
    du[2:N???+1] = dDiff_c .+ Resp     #.- u[1:N???] #.+ dDiff_u
    du[2] = du[2] - efflux
    du[1] = efflux - u[1]*unit_conv_flux
  
end

pDAMM
p = Float32.([log(6.064838*2);log(2.5);Flux.glorot_uniform(length(pML))])
problem = ODEProblem(fp,vec(u0),tspan,p)

using DiffEqSensitivity
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
function predict(p,saveat)
  solution = solve(problem
                         ,TRBDF2(autodiff = false)
                         #Tsit5()
                         #,Euler(), dt = minimum(diff(saveat))/900
                         ,u0 = u0, p=p
                         ,saveat = saveat
                         ,abstol=1e-4, reltol=1e-4
                         #,sensealg=InterpolatingAdjoint()
                         #,sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP())
                         ,sensealg=sensealg #InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
                         #,sensealg=SensitivityADPassThrough()
                         #,sensealg=QuadratureAdjoint()
                         #,sensealg = InterpolatingAdjoint(checkpointing=true)
                         #,callback=SavingCallback((u,t,integrator)-> fRespirationODE(u[1],p,t), saved_values, saveat = Float32.(saveat))
                         #,callback = SavingCallback((u,t,integrator)->integrator(t,Val{1}), saved_values, saveat = Float32.(saveat))
                         #,maxiter = 1000
                         )
    Array(solution)'
end


obs = artificial #dfk
saveat = Float32.(obs.timestep_one)
@time sol = predict(p,saveat)

# plot obs vs mod
function plot_Flux_Conc_obs_mod(solution)
  p1 = scatter(obs.co2flux_umol_m2_s1, labels = "obs       ", legend = :outertopright, title = "CO??? efflux (??mol m?? s?????)", xlab = "Months", colour = 4)
  #plot!(resDAMM, labels = "DAMM", legend = :topright)
  plot!(solution[:,1], labels = "mod       ", colour = 4)
  ppm = umolcubic_to_ppm(solution[:,[1,2,4].+1],obs.TA)

  # pick depth, unit check???
  pmod = plot(ppm,
   #title = "DAMM + Diffusion $Ds_method modelled CO??? (ppm)"
   title = "CO??? (ppm)"
   #,labels = ["5 cm" "10 cm" "20 cm"], legend = :inline)
   ,labels = ["mod 5 cm" "mod 10 cm" "mod 20 cm"], legend = :outertopright, color = [1 2 3])
  # http://painterqubits.github.io/Unitful.jl/stable/ would be cool

  #pobs = plot(obs.soilco2_5cm_ppm_PLOT, labels = "5 cm", title = "observed CO??? (ppm)", legend = :inline)
  scatter!(obs.soilco2_5cm_ppm_PLOT, labels = "obs 5 cm", color = 1)
  scatter!(obs.soilco2_10cm_ppm_PLOT, labels = "obs 10 cm", color = 2)
  scatter!(obs.soilco2_20cm_ppm_PLOT, labels = "obs 20 cm", color = 3)
  plot!(size=(1100,700))

  #plot(pmod,pobs, p1, width = 1, layout = (3,1))
  plot(pmod, p1, width = 2, layout = (2,1), markersize = 2.5)
  plot!(size=(1100,700),left_margin=5mm, bottom_margin=5mm)
end
plot_Flux_Conc_obs_mod(sol)
myplot("hybrid_with_random_parameters gelu deep")

using Statistics
function loss(p)

  pred = predict(p,Float32.(obs.timestep_one))

  mod_efflux = pred[:,1]
  mod_ppm = umolcubic_to_ppm(pred[:,[1,2,4].+1],Float32.(obs.TA))
  #mod_ppm = umolcubic_to_ppm(pred[:,[7,12,22]],Float32.(obs.TA))

  res_efflux = Flux.mse(mod_efflux,Float32.(obs.co2flux_umol_m2_s1))/mean(Float32.(obs.co2flux_umol_m2_s1))
  res_ppm_5cm = Flux.mse(mod_ppm[:,1],Float32.(obs.soilco2_5cm_ppm_PLOT))/mean(Float32.(obs.soilco2_5cm_ppm_PLOT))
  res_ppm_10cm = Flux.mse(mod_ppm[:,2],Float32.(obs.soilco2_10cm_ppm_PLOT))/mean(Float32.(obs.soilco2_10cm_ppm_PLOT))
  res_ppm_20cm = Flux.mse(mod_ppm[:,3],Float32.(obs.soilco2_20cm_ppm_PLOT))/mean(Float32.(obs.soilco2_20cm_ppm_PLOT))

  sum([res_efflux;res_ppm_5cm;res_ppm_10cm;res_ppm_20cm])

end

using Zygote

losses = []
# https://github.com/tobydriscoll/fnc-extras
callback(??,l) = begin
    push!(losses, l)
    if length(losses)%1==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
        opt_result = predict(??,Float32.(obs.timestep_one))
        plot_opt = plot_Flux_Conc_obs_mod(opt_result)
        display(plot_opt)
        #myplot("$(length(losses))")
    end
    false
end

sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))
@time loss(p)

loss(p)
ini_result = predict(p,saveat)
plot_ini = plot_Flux_Conc_obs_mod(ini_result)

# try these interpolations


# with JIT
@time sum(gradient(loss,p))
# compiled
@time sum(gradient(loss,p))

using Flux
using Optim
# The stiffness means that explicit optimizers have stiffness issues, so ADAM???s learning rate needs to be set low, 
# or use BFGS (which is what I???d recommend there instead).

# artificial data training
p_dxdt = DiffEqFlux.sciml_train(loss, p, BFGS(initial_stepnorm = 0.01), maxiters = 10, cb = callback)

p_dxdt = DiffEqFlux.sciml_train(loss, p_dxdt.minimizer, BFGS(initial_stepnorm = 0.001), maxiters = 200, cb = callback)

p_dxdt = DiffEqFlux.sciml_train(loss, p_dxdt.minimizer, BFGS(initial_stepnorm = 0.01), maxiters = 100, cb = callback)
p_dxdt = DiffEqFlux.sciml_train(loss, p_dxdt.minimizer, BFGS(initial_stepnorm = 0.01), maxiters = 100, cb = callback)

#p_dxdt = DiffEqFlux.sciml_train(loss, p_dxdt.minimizer, ADAM(0.00001), maxiters = 1000, cb = callback)

opt_result = predict(p_dxdt.minimizer,Float32.(obs.timestep_one))


plot_opt = plot_Flux_Conc_obs_mod(opt_result)
png(plot_opt, joinpath("png", "plot_opt_klappt"))

stop("training works")


#===============================================================================
# # make it work with sciml_train
===============================================================================#
# https://github.com/SciML/DiffEqFlux.jl/issues/432
# for sciml_train

#===============================================================================
# like Fabian with LSTM
===============================================================================#

struct RecoHybrid
  RbNet
  Q10
end

using Flux
@Flux.functor RecoHybrid

function (m::RecoHybrid)(x)
  Rbin,T = x
  Tref = 15.0
  rb   = m.RbNet(Rbin)
  #q10t = m.Q10[1] .^ (0.1 .* (T.-Tref))
  #reco = q10t .- rb
end

RecoHybrid(RbNet) = RecoHybrid(RbNet,log[1.5])

#mnn = Chain(LSTM(2, 16), Dense(16, 1, x -> x^2)) # 1261
#length(initial_params(mnn))

mnn = Chain(Dense(2, 16,tanh),Dense(16, 16,tanh),Dense(16, 1)) # 62
length(initial_params(mnn))

m = RecoHybrid(mnn,[1.3])

timeobs = 1:length(obs.timestep_one)

# normalization
using StatsBase
sdf = fit(UnitRangeTransform, obs.TA, dims=1)
xTA = (Float32.(StatsBase.transform(sdf, obs.TA)))*2 .- 1.f0
plot(xTA)

sdf = fit(UnitRangeTransform, obs.SWC_1, dims=1)
xSWC_1= (Float32.(StatsBase.transform(sdf, obs.SWC_1)))*2 .- 1.f0
plot(xSWC_1)

tmpConstructSequence = [xTA xSWC_1]
Rbin = [tmpConstructSequence[t,:] for t = timeobs]
T = [[obs.TA[t]] for t = timeobs]


# implement minibatches https://diffeqflux.sciml.ai/v1.24/examples/minibatch/
using Zygote
function destructure(m; cache = IdDict())
    xs = Zygote.Buffer([])
    Flux.fmap(m) do x
      if x isa AbstractArray
        push!(xs, x)
      else
        cache[x] = x
      end
      return x
    end
    return vcat(vec.(copy(xs))...), p -> _restructure(m, p, cache = cache)
  end
  
  function _restructure(m, xs; cache = IdDict())
    i = 0
    Flux.fmap(m) do x
      x isa AbstractArray || return cache[x]
      x = reshape(xs[i.+(1:length(x))], size(x))
      i += length(x)
      return x
    end
  end
  

p, re = destructure(m)

function fQ10(T;Q10,Tref = 15.0 + 273.15)
  Q10 .^ (0.1 .* (T .- Tref))
end 

function fRbQ10(Rb,q10t)
  q10t .* Rb
  #Rb .- q10t
end


function fhybrid(m::RecoHybrid,Rbin,T)
  Flux.reset!(m.RbNet)
  rb   = m.RbNet.(Rbin)
  q10t = fQ10.(T;Q10=m.Q10[1])
  #q10t = fDAMMLSTM.(T,Moistv;Q10=m.Q10[1],KmSx=m.Q10[2],KmO2=m.Q10[3])
  reco = fRbQ10.(rb,q10t)
end

res = fhybrid(m,Rbin,T)

function predict_adjoint() 
  Array(fhybrid(m,Rbin,T))
end

predict_adjoint() 

function loss_adjoint()
  pred = reduce(vcat,predict_adjoint())
  res = (Float64.(obs.co2flux_umol_m2_s1'[timeobs]) .- pred[timeobs])
  sum(abs2, res[.!isnan.(res)])/length(res[.!isnan.(res)])
end

loss_value = loss_adjoint()

using BSON: @load 
using Zygote
#@load "weights/mymodel.bson" weights
#Flux.loadparams!(m, weights)
loss_adjoint()

using BSON: @save
data = Iterators.repeated((), 100)
opt = ADAM(.001)
#opt = ADAMW()
iter = 0

thelosses = []
push!(thelosses, loss_value)
cb = function ()
global iter += 1
if iter % 20 == 0
  loss_value = loss_adjoint()
  display("After $iter iterations, the loss is $loss_value")
  theweights = params(m)
  if loss_value < minimum(thelosses)
      @save "weights/mymodel.bson" theweights
      resopt = fhybrid(m,Rbin,T)
      resopt = reduce(vcat,resopt)
  
      # initial plot with random weights and biases
      p1 = plot_Flux_LSTM_hybrid(resopt)
  
      display(p1)
  end
  push!(thelosses, loss_value)
end
end
cb()
Flux.train!(loss_adjoint, Flux.params(m), data, opt, cb = cb)
cb()

@load "weights/mymodel.bson" theweights
Flux.loadparams!(m, theweights)
loss_adjoint()

png(plot_opt, joinpath("png", "plot_lstm"))

m.Q10
resopt = fhybrid(m,Rbin,T)
resopt = reduce(vcat,resopt)





function fhybrid_params(p,Rbin,T)
    Tref = 15.0 + 273.15
    mr = re(p)
    Flux.reset!(mr.RbNet)
    rb   = mr.RbNet.(Rbin)
    q10t = fQ10.(T;Q10=exp(mr.Q10[1]))
    reco = fRbQ10.(rb,q10t)
  end


function predict_adjoint_params(p) 
    Array(fhybrid_params(p,Rbin,T))
end

predict_adjoint_params(Float32.(p))

function loss_adjoint_params(p)
    pred = reduce(vcat,predict_adjoint_params(p))
    res = (obs.co2flux_umol_m2_s1'[timeobs] .- pred[timeobs])
    sum(abs2, res[.!isnan.(res)])/length(res[.!isnan.(res)])
end

testsciml = DiffEqFlux.sciml_train(loss_adjoint_params
    ,Float32.(p)
    ,ADAM()
    ,maxiters=100
    #,allow_f_increases = true
    #,save_best = true
    ,cb = callback
    )

testsciml = DiffEqFlux.sciml_train(loss_adjoint_params
    ,Float32.([testsciml.minimizer[1:end-1];1.5])
    ,BFGS(initial_stepnorm=0.1)
    ,maxiters=10
    ,allow_f_increases = true
    #,save_best = true
    ,cb = callback
    )

res = fhybrid_params(testsciml.minimizer,Rbin,T)
res = reduce(vcat,res)
 
function plot_Flux_LSTM_hybrid(solution)
  p1 = scatter(obs.co2flux_umol_m2_s1, labels = "obs       ", legend = :outertopright, title = "CO??? efflux (??mol m?? s?????)", xlab = "Months", colour = 4)
  #plot!(resDAMM, labels = "DAMM", legend = :topright)
  plot!(solution, labels = "Hybrid-Q10-FNN       ", colour = 4)

  plot!(size=(1100,700),left_margin=5mm, bottom_margin=5mm)
end

plot_opt = plot_Flux_LSTM_hybrid(res)
png(plot_opt, joinpath("png", "plot_lstm"))

# initial plot with random weights and biases
plot(obs.co2flux_umol_m2_s1)
plot!(reduce(vcat,res), alpha = 0.7)



x_dfk = @linq dfall |> 
  select(
    #:TA,
    :SW_POT_sm, :SW_POT_sm_diff
  )
x_Rb_woTA = Array(x_dfk)'

x_Rb_woTA = Float32.(Flux.normalise(x_Rb_woTA))

plot(x_Rb_woTA')

x_dfk = @linq dfall |> 
  select(
    :TA,
    :SW_POT_sm, :SW_POT_sm_diff
  )
x_Rb_wiTA = Array(x_dfk)'
x_Rb_wiTA = Float32.(Flux.normalise(x_Rb_wiTA))

plot(x_Rb_wiTA')

# example run with random NN
NN = FastChain(FastDense(size(x_Rb_wiTA,1), 16, tanh),FastDense(16, 16, tanh),FastDense(16, 1,x->x^2))
p_NN_random = Float32.(initial_params(NN))