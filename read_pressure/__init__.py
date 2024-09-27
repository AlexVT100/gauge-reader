try:
    import os

    from homeassistant.exceptions import HomeAssistantError
    from homeassistant.helpers import device_registry as dr
    from homeassistant.helpers import entity_registry as er

    from .gauge_reader import read_pressure

    log.info('Reloading')
    logger.set_level(**{'custom_components.pyscript.apps.read_pressure': 'debug'})
    logger.set_level(**{'custom_components.pyscript.apps.read_pressure.gauge_reader': 'debug'})


    class Log:
        """
        "state", "log" and "task" are not variables but just function name prefixes (huh!)
        """
        debug = log.debug
        info = log.info
        warning = log.warning
        error = log.error


    @time_trigger('cron(* * * * *)')
    def trigger(action=None, id=None):

        # Load the configuration parameters
        camera_device = pyscript.app_config[0]['camera_device']
        image_dir = pyscript.app_config[0]['image_dir']
        gauge_image = pyscript.app_config[0]['image_file']

        # Make full paths to the required image files
        image_file_name = os.path.join(image_dir, gauge_image)

        # Take a snapshot
        try:
            camera.snapshot(entity_id=camera_device, filename=image_file_name, blocking=True)
            log.info(f'Snapshot from "{camera_device}" has been saved as "{image_file_name}"')
        except HomeAssistantError:
            return

        # Recognize the value
        value = read_pressure(Log(), image_file_name, True)
        log.info(f'Recognized pressure value: {value}')

        # Update the sensor
        sensor.boiler_pressure = value
        sensor.boiler_pressure.precision = 3
        sensor.boiler_pressure.unit_of_measurement = 'bar'
        sensor.boiler_pressure.friendly_name = 'Boiler Pressure Gauge Value'
        sensor.boiler_pressure.device_class = 'pressure'
        sensor.boiler_pressure.icon = 'mdi:gauge'

        cam_dev = get_device(hass, camera_device)
        sensor.boiler_pressure.unique_id = f'sensor.boiler_pressure.{cam_dev.name}'


    def get_device(hass, entity_id: str) -> dr.DeviceEntry:
        """
        Get device info by its entity_id

        https://community.home-assistant.io/t/how-i-can-get-deviceentry-by-entity-id-in-python/278458/2
        """
        entity_reg = er.async_get(hass)
        entry = entity_reg.async_get(entity_id)

        dev_reg = dr.async_get(hass)
        device = dev_reg.async_get(entry.device_id)

        return device

except ModuleNotFoundError:
    from .gauge_reader import read_pressure

